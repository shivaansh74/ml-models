"""
Machine Learning Models Repository

This is the main entry point for the Machine Learning Models Repository.
It provides a command-line interface to run the examples.
"""

import argparse
import sys
import os
from importlib import import_module


def get_available_models():
    """Get a list of available models in the repository."""
    models = []
    
    # Check supervised models
    supervised_dir = os.path.join('models', 'supervised')
    if os.path.exists(supervised_dir):
        for file in os.listdir(supervised_dir):
            if file.endswith('.py') and file != '__init__.py':
                model_name = file[:-3]  # Remove .py extension
                models.append(('supervised', model_name))
    
    # Check unsupervised models
    unsupervised_dir = os.path.join('models', 'unsupervised')
    if os.path.exists(unsupervised_dir):
        for file in os.listdir(unsupervised_dir):
            if file.endswith('.py') and file != '__init__.py':
                model_name = file[:-3]  # Remove .py extension
                models.append(('unsupervised', model_name))
    
    # Check ensemble models
    ensemble_dir = os.path.join('models', 'ensemble')
    if os.path.exists(ensemble_dir):
        for file in os.listdir(ensemble_dir):
            if file.endswith('.py') and file != '__init__.py':
                model_name = file[:-3]  # Remove .py extension
                models.append(('ensemble', model_name))
    
    # Check deep learning models
    deep_learning_dir = os.path.join('models', 'deep_learning')
    if os.path.exists(deep_learning_dir):
        for file in os.listdir(deep_learning_dir):
            if file.endswith('.py') and file != '__init__.py':
                model_name = file[:-3]  # Remove .py extension
                models.append(('deep_learning', model_name))
    
    return models


def get_available_examples():
    """Get a list of available examples in the repository."""
    examples = []
    
    examples_dir = 'examples'
    if os.path.exists(examples_dir):
        for file in os.listdir(examples_dir):
            if file.endswith('.py') and file != '__init__.py':
                example_name = file[:-3]  # Remove .py extension
                examples.append(example_name)
    
    return examples


def run_example(example_name):
    """Run a specific example."""
    example_path = os.path.join('examples', f'{example_name}.py')
    
    if not os.path.exists(example_path):
        print(f"Error: Example '{example_name}' not found.")
        return False
    
    print(f"Running example: {example_name}")
    print("=" * 50)
    
    # Import and run the example
    try:
        example_module = import_module(f'examples.{example_name}')
        # If the example has a main function, call it
        if hasattr(example_module, 'main'):
            example_module.main()
        else:
            # Otherwise, the example code will run on import
            pass
        
        print("\nExample completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error running example: {str(e)}")
        return False


def main():
    """Main entry point for the repository."""
    parser = argparse.ArgumentParser(description='Machine Learning Models Repository')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models and examples')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run an example')
    run_parser.add_argument('example', help='Name of the example to run')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'list':
        print("Available Models:")
        print("=" * 50)
        models = get_available_models()
        for category, model_name in models:
            print(f"- {category}/{model_name}")
        
        print("\nAvailable Examples:")
        print("=" * 50)
        examples = get_available_examples()
        for example in examples:
            print(f"- {example}")
    
    elif args.command == 'run':
        run_example(args.example)
    
    else:
        # No command provided, show help
        parser.print_help()


if __name__ == "__main__":
    main() 