from fastapi import Request

# Utility functions for users and groups
def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def load_groups():
    with open(GROUP_FILE, "r") as f:
        return json.load(f)

def save_groups(groups):
    with open(GROUP_FILE, "w") as f:
        json.dump(groups, f, indent=4)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    users = load_users()
    user = users.get(username)
    if user and user["password"] == password:
        return user
    return None

def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=403, detail="Not authenticated")
    return user

def user_in_group(username: str, group: str):
    groups = load_groups()
    return username in groups.get(group, [])