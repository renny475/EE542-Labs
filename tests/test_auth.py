from conftest import auth_headers


async def test_signup_then_login_returns_token(client):
    r = await client.post("/api/signup", json={"username": "demo", "password": "password123"})
    assert r.status_code == 200
    assert r.json()["username"] == "demo" and r.json()["token"]
    r = await client.post("/api/login", json={"username": "demo", "password": "password123"})
    assert r.status_code == 200
    token = r.json()["token"]
    me = await client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json() == {"username": "demo"}


async def test_duplicate_username_409(client):
    body = {"username": "demo", "password": "password123"}
    assert (await client.post("/api/signup", json=body)).status_code == 200
    assert (await client.post("/api/signup", json=body)).status_code == 409


async def test_short_password_422(client):
    r = await client.post("/api/signup", json={"username": "demo", "password": "short"})
    assert r.status_code == 422


async def test_wrong_password_401(client):
    await client.post("/api/signup", json={"username": "demo", "password": "password123"})
    r = await client.post("/api/login", json={"username": "demo", "password": "wrongpass1"})
    assert r.status_code == 401


async def test_protected_route_without_token_401(client):
    assert (await client.get("/api/me")).status_code == 401
    bad = await client.get("/api/me", headers={"Authorization": "Bearer garbage"})
    assert bad.status_code == 401


async def test_protected_route_with_token(client):
    headers = await auth_headers(client, "bob")
    r = await client.get("/api/me", headers=headers)
    assert r.json() == {"username": "bob"}
