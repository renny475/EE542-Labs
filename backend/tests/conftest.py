import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# settings.py refuses to import without a real JWT secret; tests must not
# depend on the developer's .env being present.
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")
