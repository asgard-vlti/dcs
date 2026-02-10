# dummy process with print statements that waits and prints extra if keyboard intterup is sent
import time
import sys


def main():
    print("Starting signit process...")
    try:
        for i in range(10):
            print(f"Signit process running... {i+1}/10")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Signit process interrupted by user.")
        sys.exit(1)
    print("Signit process completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
