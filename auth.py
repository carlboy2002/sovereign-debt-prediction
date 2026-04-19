import os
import ee

# Read project ID from environment variable or config.py
try:
    from config import GCP_PROJECT_ID
    project_id = GCP_PROJECT_ID
except ImportError:
    project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id-here")

ee.Authenticate(auth_mode="notebook", force=True)
ee.Initialize(project=project_id)
print(ee.Number(1).add(1).getInfo())