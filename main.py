import asyncio
import core as core
from dotenv import load_dotenv

def main():
    print("ChatPKD - Text-based Interface")
    print("Type your questions about Polycystic Kidney Disease.")
    print("Type 'quit' or 'exit' to close the application.")
    print("-" * 50)
    
    # Initialize the session service
    asyncio.run(core.initialize_session())

    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for using ChatPKD. Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Get response from agent
            print("ChatPKD is thinking...")
            response = asyncio.run(core.agent_response(user_input))
            print(f"ChatPKD: {response}")
            
    except KeyboardInterrupt:
        print("\n\nApplication interrupted. Closing...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        core.close_application()
        print("Application closed.")

if __name__ == "__main__":
    main()