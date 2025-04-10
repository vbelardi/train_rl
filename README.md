
# Guide d'exécution de la simulation de swarm

Ce document décrit les étapes pour exécuter la simulation du swarm dans un environnement ROS2.

---

## Prérequis

Avant de commencer, assure-toi d'avoir installé ROS2 Jazzy et configuré l'environnement Python. Ce guide part du principe que ces étapes sont déjà réalisées.


J'avais des conflits avec python et ROS2 donc j'ai crée un venv et un dossier build_rl dont il faudra exporter le PYTHONPATH

---

## 1. Accéder au workspace ROS2

Ouvre un terminal et navigue vers le répertoire de ton workspace ROS2 :

```bash
cd ~/ros2_ws
```

---

## 2. Sourcer l'environnement ROS2 et activer l'environnement virtuel

Ensuite, tu dois sourcer l'environnement ROS2 et activer l'environnement virtuel Python. Pour cela, exécute les commandes suivantes dans le terminal :

```bash
source /opt/ros/jazzy/setup.bash
source venv/bin/activate
```

Cela configure l'environnement ROS2 et active ton environnement virtuel Python pour la simulation.

---

## 3. Définir la variable d'environnement `PYTHONPATH`

Afin de garantir que Python puisse accéder aux bons modules, tu dois ajouter le répertoire `build_rl` au `PYTHONPATH` comme suit :

```bash
export PYTHONPATH=$PYTHONPATH:$PWD/build_rl
```

Cela permet de garantir que le système puisse trouver les modules Python nécessaires dans le répertoire `build_rl`.

---


## 4. Exécuter la simulation du swarm

```bash
cd build_rl/ && make -j$(nproc)
cd .. && python swarm_simulation.py
```

## Notes complémentaires

- **Assurez-vous d'être dans le bon environnement :** Si tu utilises un environnement virtuel Python, n'oublie pas de l'activer avant d'exécuter les commandes.
- **Dépendances supplémentaires :** Vérifie que toutes les dépendances nécessaires sont installées. Si tu rencontres des erreurs liées aux dépendances manquantes, consulte la documentation de ton projet ou utilise `pip install` pour installer les modules requis.
- **Si la simulation ne démarre pas :** Vérifie les messages d'erreur pour identifier si une configuration est manquante ou si une autre étape n'a pas été suivie correctement.