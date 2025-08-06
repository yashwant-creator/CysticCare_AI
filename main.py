import threading
import time
import asyncio
import core as core
from dotenv import load_dotenv



def main():
    print("Main function is running.")
    print("Starting a new thread... using threading api")

    print("ChatPKD is listening....")
    
    # Initialize the session service
    asyncio.run(core.initialize_session())

    listener = threading.Thread(target=core.background_listener, daemon=True)
    listener.start()

    print("ChatPKD stoppped listening....")
    

    try:
        while True:
            time.sleep(1) 
        
    except KeyboardInterrupt:
        core.close_application()
        print("Completely closed the application.")


if __name__ == "__main__":
    main()