"""
Garbage Segmentation App - Entry Point

Launch the Gradio web interface for garbage detection and classification.
Works both locally and on Hugging Face Spaces.
"""

import os

# Check if running on Hugging Face Spaces
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

from models.gradio_app import create_interface, launch_app

# Create the interface for HF Spaces (module-level for Gradio SDK)
demo = create_interface()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Garbage Detection Web Interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🗑️  Garbage Classifier for Waste Management")
    print("   Minor Project | UTD CSVTU Bhilai | CSE(AI) 5th Sem")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    launch_app(
        share=args.share,
        server_port=args.port,
        server_name=args.host
    )
