# Full-workspace resources (not deployed on serverless/Free targets)

These bundle resources target a **full Databricks workspace** where you can create
your own Unity Catalog, host a GPU/CPU Model Serving endpoint, and deploy a
Databricks App:

- `catalog.yml` — own catalog `baunorm_<env>`, schemas, volume, registered model.
- `serving.yml` — self-hosted `multilingual-e5-large` embedding endpoint.
- `app.yml` — the Flask Databricks App wired to the agent serving endpoint.

They are intentionally **not** in the active `include:` list in `databricks.yml`
(which only pulls `resources/jobs.yml`) because the default deploy targets a
serverless/Free workspace that uses the existing `workspace` catalog, managed
embeddings, and no custom Model Serving. To use them, add them back to `include:`
and deploy against a workspace with the matching capabilities.
