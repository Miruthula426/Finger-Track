print("==== FingerTrack AI System ====")
print("1. Collect training data (data.py)")
print("2. Predict finger-traced letter (m.py)")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    import data
elif choice == "2":
    import m
else:
    print("❌ Invalid choice")
