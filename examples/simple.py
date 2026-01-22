from pyremote import remote

x = 10
y = [1, 2, 3]

@remote("localhost", "ubuntu", password="ubuntu123", python_path="python3.10")
def compute():
    result = x * 2
    y.append(result)
    print(f"Computed: {result}")
    return result

r = compute()
print(y)  # [1, 2, 3, 20]
print(r)