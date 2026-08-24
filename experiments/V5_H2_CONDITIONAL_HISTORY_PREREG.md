# V5 H2 conditional-history preregistration

This experiment tests three fixed additions to the frozen V3 C recipe. The
features are empirical-Bayes rates built only from completed official train
seasons and looked up with values present in the current row. No evaluation
row is aggregated with another evaluation row.

- `pitcher`: pitcher × game type × count and pitcher × batter hand × count.
- `batter`: batter × count and batter × pitcher hand × count.
- `joint`: the union of those two sets.

All tables use `K=500` and all completed seasons. Candidate directions are
routed only on regular-season rows; final-type rows stay equal to V3 because
the documented 2023 label break makes older F transfer unreliable.

Only 2022 and 2023 may be read during selection. A variant/gamma is eligible
only if both point gains and both pitcher-cluster 95% lower bounds are positive.
The winner is locked by worst R gain before its 2024 prediction is generated.
The 2024 fold is confirmation only and cannot change the variant or gamma.

The Goal remains active unless the V5 conservative expected-score lower bound
exceeds 1190 or a manually submitted candidate records an actual DACON score
above 1190.
