"""
One-time setup: store the Lakebase connection URL in a Databricks secret scope.

Run once from a Databricks notebook (or locally with the Databricks CLI/SDK
configured):
    python setup_secrets.py

Uses getpass so the URL is never echoed to the screen or written to shell
history. Unlike the Massive reference app, baunorm-finder has no external API
key - the only secret is the Lakebase URL.
"""
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

w.secrets.create_scope(scope="database")
w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: "),
)

# Allow the app (running as a workspace user) to read the secret at runtime.
w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)

print("OK - stored database/lakebase-url")
