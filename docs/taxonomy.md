# Taxonomy

The controlled vocabulary lives in `storygraph.understanding.taxonomy`. The LLM may emit free
text; `Taxonomy.snap` maps it to these terms (exact, underscore / space variants, synonym table,
then loose containment) and drops anything that does not resolve.

## Genres (max 3 per title)

literary_fiction, crime, thriller, mystery, horror, science_fiction, fantasy, romance,
historical_fiction, young_adult, memoir, biography, history, science, technology, business,
economics, politics, true_crime, self_help, travel, nature, philosophy, religion, comedy,
interview, news, sports, health, education

## Themes (max 8 per title, kept if present in >= 15 % of words)

grief, family, isolation, identity, memory, betrayal, redemption, survival, coming_of_age, power,
class, money, trust, faith, war, migration, nature, technology, justice, love, friendship,
ambition, loss, secrets, the_sea, community, work, madness, the_supernatural, fear,
coordination_failure, institutions, language, home, freedom

## Moods (max 4, ordered as a trajectory through the title)

eerie, melancholic, tense, hopeful, warm, wry, urgent, contemplative, playful, bleak, cozy,
analytical, suspenseful, nostalgic, angry, serene

## Narrative arcs

rags_to_riches, riches_to_rags, man_in_hole, icarus, cinderella, oedipus, episodic, expository

Arcs are classified from the 12-bin tension curve by correlation against templates; curves with
standard deviation under 0.05 are `expository`, and correlations under 0.5 fall back to `episodic`.

## Editorial agreement

`storygraph.evaluation.agreement` reports exact-match, macro-F1, micro-F1 and Cohen's kappa per
field against editor gold labels. The tagging config targets 0.90 macro-F1 on genres before an
adapter version is promoted.
