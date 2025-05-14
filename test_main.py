from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app, Base, get_db
from main import get_password_hash, User

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the get_db dependency to use the test database
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create the test database
Base.metadata.drop_all(bind=engine)  # Clear the database before creating tables
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def setup_test_user():
    """
    Helper function to create a test user in the database.
    """
    db = TestingSessionLocal()
    hashed_password = get_password_hash("testpassword")
    test_user = User(
        username="testuser",
        email="testuser@example.com",
        hashed_password=hashed_password,
        is_active=True,
        is_admin=False
    )
    db.add(test_user)
    db.commit()
    db.close()

def test_login_for_access_token():
    """
    Test the /token endpoint for successful login.
    """
    # Setup test user
    setup_test_user()

    # Send POST request to /token
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "testpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Assert the response
    assert response.status_code == 200
    response_data = response.json()
    assert "access_token" in response_data
    assert response_data["token_type"] == "bearer"
    assert response_data["username"] == "testuser"
    assert response_data["is_admin"] is False

def test_login_with_invalid_credentials():
    """
    Test the /token endpoint with invalid credentials.
    """
    # Send POST request to /token with invalid credentials
    response = client.post(
        "/token",
        data={"username": "invaliduser", "password": "wrongpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Assert the response
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password"}

def test_login_with_inactive_user():
    """
    Test the /token endpoint with an inactive user.
    """
    # Create an inactive user
    db = TestingSessionLocal()
    hashed_password = get_password_hash("inactivepassword")
    inactive_user = User(
        username="inactiveuser",
        email="inactiveuser@example.com",
        hashed_password=hashed_password,
        is_active=False,
        is_admin=False
    )
    db.add(inactive_user)
    db.commit()
    db.close()

    # Send POST request to /token
    response = client.post(
        "/token",
        data={"username": "inactiveuser", "password": "inactivepassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Assert the response
    assert response.status_code == 403
    assert response.json() == {"detail": "User account is inactive"}