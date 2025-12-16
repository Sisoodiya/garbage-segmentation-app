"""
Garbage Segmentation App - Entry Point

Launch the Gradio web interface for garbage detection and classification.
"""

from models.gradio_app import launch_app

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
