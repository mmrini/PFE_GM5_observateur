# Projet de fin d'études : observateur pour les équations des ondes appliqué à l'imagerie médicale

*Réalisé par :* Malak Mrini

*Sous la supervision de :* Dr. Antoine Tonnoir 

## Description

La propagation d’ondes dans un milieu est au cœur de nombreuses applications scientifiques, notamment en imagerie médicale, et en particulier en élastographie, une technique qui exploite la propagation d’ondes mécaniques pour caractériser les propriétés des tissus biologiques. Dans ce contexte, la dynamique du système peut être modélisée par l’équation des ondes. Une difficulté importante consiste alors à reconstruire l’état du système à partir d’un nombre limité de mesures, ce qui conduit à un problème inverse.

L’objectif de ce projet est d’étudier la reconstruction des conditions initiales de l’équation des ondes à partir de mesures partielles de vitesse disponibles uniquement dans une zone d’observation restreinte. Pour traiter ce problème, nous utilisons une approche fondée sur la théorie des observateurs pour les équations aux dérivées partielles, qui permet d’estimer progressivement l’état du système en corrigeant la dynamique du modèle à partir des observations.

La reconstruction est réalisée grâce à la méthode Back and Forth Nudging (BFN), qui alterne des phases de propagation directe et rétrograde afin de ramener l’information vers l’instant initial. L’équation des ondes est discrétisée à l’aide d’un schéma explicite centré d’ordre deux en temps et en espace, permettant la mise en œuvre numérique de l’observateur et de l’algorithme.

Des simulations numériques sont ensuite réalisées afin d’analyser l’influence du paramètre de nudging et de la géométrie de la zone d’observation sur la qualité de la reconstruction. Les résultats mettent en évidence l’importance de la configuration des observations pour la convergence de l’algorithme et illustrent l’intérêt des méthodes d’observation pour la résolution de problèmes inverses liés à la propagation d’ondes.

## Utilisation 

Exemple de commande : 
`python run_bfn_experiments.py --masks two_walls_l,thick_border,circle_center,circle_excenter --ratios 0.3 --gammas 1.0,5.0,10.0 --bfn_iters 15 --verbose`

## Visualisation et sauvegarde
Les sorties sont enregistrées dans `data/`.
