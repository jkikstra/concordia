# Alternative simplified version for Dask setup
# This can be used as a replacement for the conditional setup in the workflow

def setup_dask_client():
    """
    Set up Dask client with environment-specific configuration.
    
    Returns:
        client: Dask client object or None for synchronous execution
    """
    import os
    import sys
    import dask.config
    from dask.distributed import Client
    
    # Check if running in VS Code interactive window
    is_vscode_interactive = (
        'vscode' in os.environ.get('TERM_PROGRAM', '').lower() or
        'VSCODE_PID' in os.environ or
        hasattr(sys, 'ps1') and 'ipykernel' in sys.modules and 
        'IPython' in sys.modules
    )
    
    # Check if running in Jupyter Lab/Notebook
    is_jupyter = (
        'ipykernel' in sys.modules and 
        'jupyter' in os.environ.get('JPY_PARENT_PID', '') or
        'jupyter-lab' in str(sys.argv[0]) or
        'jupyter-notebook' in str(sys.argv[0])
    )
    
    try:
        if is_vscode_interactive:
            print("Detected VS Code interactive environment - using threaded scheduler")
            # Use threaded scheduler for VS Code interactive window
            dask.config.set(scheduler='threads')
            client = None  # Use synchronous/threaded execution
            
        elif is_jupyter:
            print("Detected Jupyter environment - using distributed client")
            # Use distributed client for Jupyter
            client = Client()
            # client.register_plugin(DaskSetWorkerLoglevel(logger().getEffectiveLevel()))
            client.forward_logging()
            import dask.distributed
            dask.distributed.gc.disable_gc_diagnosis()
            
        else:
            print("Detected script/other environment - attempting distributed client")
            # Try distributed client for other environments
            try:
                client = Client()
                # client.register_plugin(DaskSetWorkerLoglevel(logger().getEffectiveLevel()))
                client.forward_logging()
                import dask.distributed
                dask.distributed.gc.disable_gc_diagnosis()
                print("Successfully created distributed client")
            except Exception as e:
                print(f"Failed to create distributed client: {e}")
                print("Falling back to threaded scheduler")
                dask.config.set(scheduler='threads')
                client = None
                
    except Exception as e:
        print(f"Error setting up Dask: {e}")
        print("Falling back to synchronous execution")
        dask.config.set(scheduler='synchronous')
        client = None
    
    return client

def setup_dask_simple():
    """
    Simplified Dask setup that tries distributed first, falls back to threads.
    """
    import dask.config
    from dask.distributed import Client
    
    try:
        # Try to create a distributed client
        client = Client()
        client.forward_logging()
        print(f"✓ Using distributed Dask client: {client.dashboard_link}")
        return client
    except Exception as e:
        # Fall back to threaded scheduler
        print(f"⚠ Distributed client failed ({e}), using threaded scheduler")
        dask.config.set(scheduler='threads')
        return None

def setup_dask_manual_override(use_distributed=None):
    """
    Manual override version - you can explicitly choose the scheduler.
    
    Args:
        use_distributed (bool): 
            - True: Force distributed client
            - False: Force threaded scheduler  
            - None: Auto-detect (try distributed, fall back to threads)
    """
    import dask.config
    from dask.distributed import Client
    
    if use_distributed is False:
        print("🧵 Manually configured for threaded scheduler")
        dask.config.set(scheduler='threads')
        return None
    
    elif use_distributed is True:
        print("🌐 Manually configured for distributed client")
        client = Client()
        client.forward_logging()
        return client
    
    else:
        # Auto-detect (same as setup_dask_simple)
        return setup_dask_simple()

# Usage examples:
# client = setup_dask_client()                             # Auto-detect environment
# client = setup_dask_simple()                             # Auto-detect (try distributed, fall back to threads)
# client = setup_dask_manual_override(use_distributed=False)  # Force threads
# client = setup_dask_manual_override(use_distributed=True)   # Force distributed