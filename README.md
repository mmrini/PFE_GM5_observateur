# Projet de fin d'études : observateur pour les équations des ondes appliqué à l'imagerie médicale

*Réalisé par :* Malak Mrini

*Sous la supervision de :* Dr. Antoine Tonnoir 

## Description

Ce dépôt à pour but d'implémenter un solveur direct aux différences finies pour l'équation des ondes et des observateurs associés pour estimer/observer le champ (et sa vitesse) à partir de mesures partielles (dans une zone d'observation D_obs). 

## Fonctionnalités principales

- Solveur direct explicite de l'équation des ondes (schéma aux différences finies).
- Opérateur de mesure (extraction de $\partial_t u$ sur une zone d'observation).
- Observateur direct (injection de correction basée sur la mesure).
- > TO DO : Implémentation de l'observateur rétrograde

## Reproduction 

Sur le termminal, taper : `python main_simulation.py`

## Visualisation et sauvegarde
Les sorties sont enregistrées dans `data/`.
