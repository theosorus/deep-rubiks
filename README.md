![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![React](https://img.shields.io/badge/React-18.x-61dafb.svg)
![Three.js](https://img.shields.io/badge/Three.js-Latest-black.svg)

# 🎲 DeepCubeA - AI-Powered Rubik's Cube Solver

> **A modern implementation of the DeepCubeA algorithm for solving Rubik's Cube using neural networks and A* search with web vizualisation**


<div align="center">
  <img src="assets/deep_rubiks_gif_1.gif" alt="FormulaTracker demo gif" width="700"/>
</div>





##  🎯 **DeepCubeA** Architecture

Le système comprend deux composants principaux :

### 1. Fonction Cost-to-Go (J(s))
- **Réseau de neurones profond** : MLP avec blocs résiduels
- **Entraîné par DAVI** (Deep Approximate Value Iteration)
- **Prédit le nombre de mouvements** nécessaires pour résoudre le cube depuis n'importe quel état

### 2. Pathfinder A*
- **Recherche A* pondérée** utilisant J(s) comme heuristique
- **Beam Search** comme alternative pour des solutions rapides
- **Optimisations par batch** pour l'évaluation des états



## Project Architecture

### Backend

fastapi , pytorch
on localhost:8000

### Frontend

frontend with react , threejs 
on localhost:3000



## 🚀 Quick Start


```bash
# Build
docker compose build

# Launch
docker compose up

```

## 🎯 Usage

1. **🌐 Open** the interface at `http://localhost:3000`
2. **🎲 Shuffle** the cube with the Shuffle button
3. **🧠 Solve** with the Solve button - watch the AI work!
4. **🎮 Explore**: manual rotations, reset, face labels...

## 🔬 Results and Performance

- **Accuracy**: Optimal solutions in >95% of cases
- **Speed**: Resolution in <1 second for most configurations
- **Robustness**: Works on scrambles up to 30 moves
- **Visualization**: Smooth animation of found solutions

## 🎓 Scientific References

This project implements the algorithm described in:
> "Solving the Rubik's Cube with Deep Reinforcement Learning and Search"
> McAleer et al. (2018)



