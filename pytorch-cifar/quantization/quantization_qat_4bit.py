# ============================================================================
# IMPORTS - Bibliothèques nécessaires pour la quantization-aware training (QAT)
# ============================================================================
import argparse  # Pour parser les arguments en ligne de commande
import copy  # Pour faire des copies profondes de modèles
import json  # Pour sauvegarder les résultats en JSON
import sys  # Pour manipuler les chemins système
from pathlib import Path  # Pour gérer les chemins de fichiers

import torch  # Framework PyTorch principal
import torch.nn as nn  # Modules de réseaux de neurones
import torch.nn.functional as F  # Fonctions au niveau de la couche
import torch.optim as optim  # Optimiseurs (SGD, Adam, etc.)
import torchvision  # Datasets et utilitaires de vision
import torchvision.transforms as transforms  # Transformations d'images

# Déterminer le répertoire racine du projet (pytorch-cifar)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importer le modèle ResNet18 depuis le dossier models
from models import ResNet18

# Chemins par défaut pour les poids, outputs et résultats
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "Models Weights" / "unstructured_pruning"  # Dossier input avec poids FP32
DEFAULT_INT4_DIR = PROJECT_ROOT / "Models Weights" / "quantization" / "unstructured_pruning_qat_4bit"  # Dossier output pour modèles quantifiés
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "quantization" / "unstructured_pruning_qat_4bit_results.json"  # Fichier JSON avec résultats

# ============================================================================
# SECTION 1: QUANTIZATION FAKE - Quantization-Aware Training avec STE
# ============================================================================
# STE = Straight-Through Estimator: une technique qui permet aux gradients
# de "passer au travers" de la fonction d'arrondi lors du backprop.
# Cela aide le réseau à apprendre les poids malgré la quantization.

class RoundSTE(torch.autograd.Function):
    """Fonction d'arrondi avec Straight-Through Estimator (STE).
    
    Forward: arrondit les valeurs à l'entier le plus proche.
    Backward: ignore l'arrondi et récupère les gradients directement.
    """
    @staticmethod
    def forward(ctx, x):
        # Forward: arrondir les valeurs
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: passer les gradients directement (ignorer l'arrondi)
        # C'est le "Straight-Through Estimator"
        return grad_output

def fake_quantize(tensor, num_bits=4, symmetric=True):
    """Simule la quantization sans vraiment quantifier le modèle.
    
    Args:
        tensor: valeurs à quantifier (ex: poids ou activations)
        num_bits: entre 4 (4-bit) ou 8 (8-bit). Ici: 4 bits
        symmetric: True pour poids (signed: -7 à 7), False pour activations (unsigned: 0 à 15)
    
    Retourne: tensor simulé quantifié et dé-quantifié (FP32 encore)
    """
    if symmetric:
        # Mode SYMÉTRIQUE pour poids: valeurs signées
        # 4-bit signé: -8 à 7 (mais on utilise -7 à 7 souvent)
        qmax = (2 ** (num_bits - 1)) - 1  # Pour 4 bits: 7
        qmin = -(2 ** (num_bits - 1))      # Pour 4 bits: -8
        
        # Trouver la plus grande valeur absolue (en valeur absolue) pour l'échelle
        max_val = tensor.abs().max().clamp(min=1e-5)  # min=1e-5 pour éviter division par 0
        scale = max_val / qmax  # Facteur d'échelle
        
        # ÉTAPE 1: Quantification (arrondir à l'entier le plus proche avec STE)
        q_tensor = RoundSTE.apply(tensor / scale)  # Diviser par scale, puis arrondir
        # ÉTAPE 2: Clamp entre qmin et qmax pour respecter la plage 4-bit
        q_tensor = torch.clamp(q_tensor, qmin, qmax)
        # ÉTAPE 3: Dé-quantification (revenir à FP32 en multipliant par scale)
        return q_tensor * scale
    else:
        # Mode ASYMÉTRIQUE pour activations: valeurs non-signées
        # 4-bit non-signé: 0 à 15
        qmax = (2 ** num_bits) - 1      # Pour 4 bits: 15
        qmin = 0
        
        # Trouver min et max des activations
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Calculer l'échelle (plage / nombre de niveaux)
        scale = (max_val - min_val).clamp(min=1e-5) / qmax
        
        # ÉTAPE 1: Normaliser les valeurs entre 0 et 1
        q_tensor = RoundSTE.apply((tensor - min_val) / scale)
        # ÉTAPE 2: Clamp entre 0 et 15
        q_tensor = torch.clamp(q_tensor, qmin, qmax)
        # ÉTAPE 3: Dénormaliser pour revenir à la plage originale
        return q_tensor * scale + min_val

class QATConv2d(nn.Conv2d):
    """Couche convolution avec quantization fake 4-bit.
    
    Quantifie les activations (asymétriques) et poids (symétriques) en fake 4-bit.
    Le STE permet à la rétropropagation de fonctionner malgré la quantization.
    """
    def forward(self, input):
        # Quantifier les activations en 4-bit ASYMÉTRIQUE (0-15)
        quantized_input = fake_quantize(input, num_bits=4, symmetric=False)
        # Quantifier les poids en 4-bit SYMÉTRIQUE (-8 à 7)
        quantized_weight = fake_quantize(self.weight, num_bits=4, symmetric=True)
        
        # Faire la convolution avec les valeurs quantifiées (mais toujours en FP32)
        return F.conv2d(
            quantized_input, quantized_weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )

class QATLinear(nn.Linear):
    """Couche fully-connected avec quantization fake 4-bit."""
    def forward(self, input):
        # Quantifier les activations en 4-bit ASYMÉTRIQUE
        quantized_input = fake_quantize(input, num_bits=4, symmetric=False)
        # Quantifier les poids en 4-bit SYMÉTRIQUE
        quantized_weight = fake_quantize(self.weight, num_bits=4, symmetric=True)
        
        # Faire la multiplication matricielle avec les valeurs quantifiées
        return F.linear(quantized_input, quantized_weight, self.bias)

def replace_layers_with_qat(model):
    """Remplace RÉCURSIVEMENT toutes les couches Conv2d et Linear par leurs versions QAT 4-bit.
    
    Cela transforme un modèle FP32 normal en modèle QAT (qui quantifie en fake 4-bit).
    Les poids et biais sont copiés de l'ancien modèle au nouveau.
    """
    # Itérer sur les sous-modules du modèle (en ordre inverse)
    for name, module in reversed(model._modules.items()):
        # Si ce module a des enfants (nested), faire la remplace récursive d'abord
        if len(list(module.children())) > 0:
            replace_layers_with_qat(module)
        
        # REMPLACER LES COUCHES CONVOLUTION
        if isinstance(module, nn.Conv2d):
            # Créer une nouvelle couche QATConv2d avec les mêmes paramètres
            qat_conv = QATConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation, module.groups,
                module.bias is not None, module.padding_mode
            )
            # Copier les poids et bias de l'ancienne couche
            qat_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_conv.bias.data = module.bias.data.clone()
            # Remplacer la couche dans le modèle
            model._modules[name] = qat_conv
            
        # REMPLACER LES COUCHES FULLY-CONNECTED
        elif isinstance(module, nn.Linear):
            # Créer une nouvelle couche QATLinear avec les mêmes paramètres
            qat_linear = QATLinear(
                module.in_features, module.out_features, module.bias is not None
            )
            # Copier les poids et bias
            qat_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_linear.bias.data = module.bias.data.clone()
            # Remplacer la couche dans le modèle
            model._modules[name] = qat_linear
    
    return model

# ---------------------------------------------------------------------------
# 2. Standard Training & Evaluation Utilities
# ---------------------------------------------------------------------------

# ============================================================================
# SECTION 2: UTILITAIRES POUR TRAINING & EVALUATION
# ============================================================================

def parse_args():
    """Analyser les arguments de la ligne de commande.
    
    Permet à l'utilisateur de customiser:
    - Les répertoires d'input/output
    - Le nombre d'epochs
    - Le learning rate
    - Les hyperparams (momentum, weight decay)
    - Le device (GPU ou CPU)
    """
    parser = argparse.ArgumentParser(description="Reliable 4-bit QAT using STE.")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR, 
                        help="Répertoire avec modèles FP32 à quantifier")
    parser.add_argument("--glob", type=str, default="*.pth", 
                        help="Pattern pour trouver les checkpoints (ex: *.pth)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INT4_DIR,
                        help="Répertoire pour sauvegarder les modèles quantifiés")
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON,
                        help="Fichier JSON pour sauvegarder les résultats")
    parser.add_argument("--batch-size-train", type=int, default=128,
                        help="Batch size pour l'entraînement")
    parser.add_argument("--batch-size-test", type=int, default=100,
                        help="Batch size pour les tests")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Nombre de workers pour data loading")
    parser.add_argument("--qat-epochs", type=int, default=10,
                        help="Nombre d'epochs pour QAT fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate pour l'optimiseur SGD")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum pour SGD")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay (L2 regularization) pour SGD")
    parser.add_argument("--train-device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device pour training (cuda ou cpu)")
    return parser.parse_args()

def build_data_loaders(batch_size_train=128, batch_size_test=100, num_workers=2):
    """Construire les data loaders pour CIFAR-10.
    
    Retourne: (train_loader, test_loader)
    """
    # TRANSFORMATIONS POUR TRAINING (avec augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Crop aléatoire + padding
        transforms.RandomHorizontalFlip(),      # Flip horizontal aléatoire
        transforms.ToTensor(),                  # Convertir en tenseur (0-1)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normaliser
    ])
    
    # TRANSFORMATIONS POUR TEST (sans augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Charger les datasets CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def load_checkpoint_to_model(checkpoint_path):
    """Charger un checkpoint FP32 dans un modèle ResNet18.
    
    Gestion de la compatibilité DataParallel (module. prefix).
    """
    model = ResNet18()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extraire state_dict (peut être dans checkpoint["net"] ou directement checkpoint)
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    
    # Supprimer le prefix "module." si présent (vient de DataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Charger dans le modèle
    model.load_state_dict(state_dict, strict=False)
    return model

def evaluate_model(model, data_loader, device):
    """Evaluer le modèle sur un dataset.
    
    Retourne: (loss_moyenne, accuracy_moyenne)
    """
    # Créer la loss function
    criterion = nn.CrossEntropyLoss()
    
    # Passer en mode évaluation (désactive dropout, batch norm se gèle, etc.)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    # No_grad: désactiver le calcul des gradients (on évalue, on n'entraîne pas)
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Déplacer les données sur le device (GPU ou CPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Accumuler les losses et accuracies
            total_loss += loss.item()
            _, predicted = outputs.max(1)  # Prendre la classe avec la plus haute probabilité
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Retourner moyenne
    return total_loss / max(1, len(data_loader)), 100.0 * correct / max(1, total)

# ============================================================================
# SECTION 3: QAT FINE-TUNING - Entraînement du modèle quantifié
# ============================================================================

def qat_finetune_4bit(fp32_model, train_loader, test_loader, train_device, args):
    """Fine-tuner un modèle FP32 avec quantization-aware training 4-bit.
    
    Processus:
    1. Faire une copie du modèle FP32
    2. Remplacer ses couches par des versions QAT (fake quantize)
    3. Entraîner en utilisant la méthode QAT (quantize -> backprop avec STE)
    4. Sauvegarder le meilleur checkpoint (selon accuracy test)
    
    Retourne: (qat_model quantifié, best_accuracy)
    """
    # ÉTAPE 1: Copier le modèle FP32 et lui ajouter les couches QAT
    qat_model = copy.deepcopy(fp32_model)
    qat_model = replace_layers_with_qat(qat_model).to(train_device)
    
    # ÉTAPE 2: Créer loss function, optimiseur et scheduler
    criterion = nn.CrossEntropyLoss()  # Loss pour classification
    optimizer = optim.SGD(qat_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.qat_epochs))

    best_acc = -1.0  # Meilleure accuracy vue jusqu'ici
    best_state = None  # State dict du meilleur modèle

    # ÉTAPE 3: BOUCLE D'ENTRAîNEMENT
    for epoch in range(args.qat_epochs):
        qat_model.train()  # Passer en mode training
        train_loss, correct, total = 0.0, 0, 0
        
        # Itérer sur les batches d'entraînement
        for inputs, targets in train_loader:
            # Déplacer données sur le device
            inputs, targets = inputs.to(device=train_device), targets.to(device=train_device)
            
            # Forward pass: inputs et poids sont quantifiés en fake 4-bit
            optimizer.zero_grad()
            outputs = qat_model(inputs)  # Passe avant avec quantization
            loss = criterion(outputs, targets)
            
            # Backward pass avec STE (le gradient ignore l'arrondi)
            loss.backward()
            optimizer.step()  # Mettre à jour les poids
            
            # Accumuler la loss et l'accuracy pour ce batch
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Ajuster le learning rate avec le scheduler (cosine annealing)
        scheduler.step()
        
        # ÉTAPE 4: EVALUER sur le test set
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate_model(qat_model, test_loader, train_device)
        
        # Afficher les résultats de cet epoch
        print(f"    Epoch {epoch + 1}/{args.qat_epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        # Sauvegarder le meilleur modèle (selon test accuracy)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(qat_model.state_dict())

    # ÉTAPE 5: Charger le meilleur modèle
    if best_state is not None:
        qat_model.load_state_dict(best_state)

    return qat_model, best_acc

def main():
    """Fonction principale: quantizer en masse des checkpoints FP32."""
    # Analyser les arguments
    args = parse_args()
    
    # Créer les répertoires d'output s'ils n'existent pas
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Charger les data loaders CIFAR-10
    train_loader, test_loader = build_data_loaders(args.batch_size_train, args.batch_size_test, args.num_workers)

    # List pour collectionner les résultats
    results = []
    
    # Trouver tous les checkpoints FP32 dans le dossier input
    checkpoint_paths = sorted(args.weights_dir.glob(args.glob))
    
    # BOUCLE PRINCIPALE: traiter chaque checkpoint
    for ckpt_path in checkpoint_paths:
        print(f"\n==> Processing: {ckpt_path.name}")
        
        # CHARGER le modèle FP32 depuis le checkpoint
        fp32_model = load_checkpoint_to_model(ckpt_path).to(args.train_device)
        
        # ÉVALUER le modèle FP32 baseline (avant quantization)
        fp32_loss, fp32_acc = evaluate_model(fp32_model, test_loader, args.train_device)
        print(f"  FP32 Test -> loss: {fp32_loss:.4f}, acc: {fp32_acc:.2f}%")

        # FINE-TUNER avec QAT 4-bit
        qat_model, qat_acc = qat_finetune_4bit(fp32_model, train_loader, test_loader, args.train_device, args)
        
        # SAUVEGARDER le modèle quantifié
        save_name = ckpt_path.stem + "_int4_qat.pth"  # Ajouter suffix "_int4_qat"
        save_path = args.output_dir / save_name
        
        torch.save({
            "net": qat_model.state_dict(),  # State du modèle QAT
            "source_checkpoint": str(ckpt_path),  # Origine du checkpoint
            "quantization": "Custom STE 4-bit",  # Méthode de quantization
            "qat_epochs": args.qat_epochs,  # Nombre d'epochs d'entraînement
            "fp32_test_acc": fp32_acc,  # Accuracy FP32 baseline
            "int4_test_acc": qat_acc,  # Accuracy après QAT
        }, save_path)
        
        print(f"  Saved to: {save_path}")
        
        # SAUVEGARDER les résultats dans la liste
        results.append({
            "checkpoint": ckpt_path.name,
            "fp32_test_acc": round(fp32_acc, 4),  # Arrondir à 4 décimales
            "int4_test_acc": round(qat_acc, 4),
            "acc_drop": round(fp32_acc - qat_acc, 4),  # Difference entre FP32 et QAT
        })

    # SAUVEGARDER les résultats finaux en JSON
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to: {args.results_json}")

# ============================================================================
# ENTRY POINT - Point d'entrée du script
# ============================================================================

if __name__ == "__main__":
    main()