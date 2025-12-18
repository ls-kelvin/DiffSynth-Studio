import torch
import sys
from typing import Dict, Any, Union
from collections import OrderedDict


def get_size_bytes(obj: Any) -> int:
    """
    Calculate memory size of an object in bytes.
    Handles tensors, lists, dicts, strings, and other common Python objects.
    """
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, (list, tuple)):
        return sum(get_size_bytes(item) for item in obj) + sys.getsizeof(obj)
    elif isinstance(obj, dict):
        return sum(get_size_bytes(k) + get_size_bytes(v) for k, v in obj.items()) + sys.getsizeof(obj)
    elif isinstance(obj, str):
        return sys.getsizeof(obj)
    elif isinstance(obj, (int, float, bool)):
        return sys.getsizeof(obj)
    elif obj is None:
        return sys.getsizeof(obj)
    else:
        try:
            return sys.getsizeof(obj)
        except:
            return 0


def format_size(size_bytes: int) -> str:
    """
    Format byte size into human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get detailed information about a tensor.
    """
    return {
        'type': 'torch.Tensor',
        'dtype': str(tensor.dtype),
        'shape': tuple(tensor.shape),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'size_bytes': tensor.element_size() * tensor.nelement(),
        'size_formatted': format_size(tensor.element_size() * tensor.nelement())
    }


def get_object_info(obj: Any, var_name: str = '') -> Dict[str, Any]:
    """
    Get information about any object.
    """
    if isinstance(obj, torch.Tensor):
        return get_tensor_info(obj)
    else:
        size_bytes = get_size_bytes(obj)
        return {
            'type': type(obj).__name__,
            'value_preview': str(obj)[:100] if not isinstance(obj, (dict, list)) else f'{type(obj).__name__} with {len(obj)} items',
            'size_bytes': size_bytes,
            'size_formatted': format_size(size_bytes)
        }


def analyze_pth_file(pth_file_path: str, max_depth: int = 3, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze a .pth file and return detailed memory information for each variable.
    
    Args:
        pth_file_path: Path to the .pth file
        max_depth: Maximum depth to analyze nested structures (default: 3)
        verbose: Whether to print detailed analysis (default: True)
    
    Returns:
        Dictionary containing analysis results with the following structure:
        {
            'total_size_bytes': int,
            'total_size_formatted': str,
            'variables': OrderedDict with variable details
        }
    """
    # Load the .pth file
    try:
        data = torch.load(pth_file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return {
            'error': f'Failed to load file: {str(e)}',
            'file_path': pth_file_path
        }
    
    results = OrderedDict()
    total_size = 0
    
    def analyze_recursive(obj: Any, name: str, depth: int = 0):
        nonlocal total_size
        
        if depth > max_depth:
            return
        
        if isinstance(obj, dict):
            dict_size = 0
            dict_info = OrderedDict()
            for key, value in obj.items():
                key_name = f"{name}.{key}" if name else str(key)
                
                if isinstance(value, torch.Tensor):
                    info = get_tensor_info(value)
                    dict_info[key] = info
                    dict_size += info['size_bytes']
                elif isinstance(value, dict):
                    analyze_recursive(value, key_name, depth + 1)
                    size = get_size_bytes(value)
                    dict_info[key] = {
                        'type': 'dict',
                        'num_items': len(value),
                        'size_bytes': size,
                        'size_formatted': format_size(size)
                    }
                    dict_size += size
                elif isinstance(value, (list, tuple)):
                    # Recursively analyze list/tuple contents
                    analyze_recursive(value, key_name, depth + 1)
                    size = get_size_bytes(value)
                    dict_size += size
                else:
                    info = get_object_info(value, key_name)
                    dict_info[key] = info
                    dict_size += info['size_bytes']
            
            if name:
                results[name] = {
                    'type': 'dict',
                    'num_items': len(obj),
                    'size_bytes': dict_size,
                    'size_formatted': format_size(dict_size),
                    'contents': dict_info
                }
            else:
                results.update(dict_info)
            
            total_size += dict_size
        
        elif isinstance(obj, (list, tuple)):
            list_size = 0
            list_info = OrderedDict()
            
            for idx, item in enumerate(obj):
                item_name = f"{name}[{idx}]" if name else f"[{idx}]"
                
                if isinstance(item, torch.Tensor):
                    info = get_tensor_info(item)
                    list_info[f"[{idx}]"] = info
                    list_size += info['size_bytes']
                elif isinstance(item, dict):
                    analyze_recursive(item, item_name, depth + 1)
                    size = get_size_bytes(item)
                    list_size += size
                elif isinstance(item, (list, tuple)):
                    analyze_recursive(item, item_name, depth + 1)
                    size = get_size_bytes(item)
                    list_size += size
                else:
                    info = get_object_info(item, item_name)
                    list_info[f"[{idx}]"] = info
                    list_size += info['size_bytes']
            
            if name:
                results[name] = {
                    'type': type(obj).__name__,
                    'num_items': len(obj),
                    'size_bytes': list_size,
                    'size_formatted': format_size(list_size),
                    'contents': list_info
                }
            else:
                # Top-level tuple/list - expand its contents
                results.update(list_info)
            
            total_size += list_size
            
        elif isinstance(obj, torch.Tensor):
            info = get_tensor_info(obj)
            results[name] = info
            total_size += info['size_bytes']
        else:
            info = get_object_info(obj, name)
            results[name] = info
            total_size += info['size_bytes']
    
    # Start analysis
    analyze_recursive(data, '')
    
    # Prepare final results
    analysis_results = {
        'file_path': pth_file_path,
        'total_size_bytes': total_size,
        'total_size_formatted': format_size(total_size),
        'num_variables': len(results),
        'variables': results
    }
    
    # Print detailed analysis if verbose
    if verbose:
        print(f"\n{'='*80}")
        print(f"PTH File Analysis: {pth_file_path}")
        print(f"{'='*80}")
        print(f"Total Size: {analysis_results['total_size_formatted']} ({total_size:,} bytes)")
        print(f"Number of Variables: {analysis_results['num_variables']}")
        print(f"\n{'-'*80}")
        print(f"{'Variable Name':<40} {'Type':<20} {'Size':<15}")
        print(f"{'-'*80}")
        
        def print_variable(var_name, var_info, indent=0):
            indent_str = "  " * indent
            
            if isinstance(var_info, dict) and 'contents' in var_info:
                # Nested dict
                print(f"{indent_str}{var_name:<{40-indent*2}} {var_info['type']:<20} {var_info['size_formatted']:<15}")
                for sub_name, sub_info in var_info['contents'].items():
                    print_variable(sub_name, sub_info, indent + 1)
            elif isinstance(var_info, dict) and 'type' in var_info:
                # Regular variable
                type_str = var_info['type']
                if type_str == 'torch.Tensor' and 'shape' in var_info:
                    type_str = f"Tensor{var_info['shape']}"
                size_str = var_info.get('size_formatted', 'N/A')
                print(f"{indent_str}{var_name:<{40-indent*2}} {type_str:<20} {size_str:<15}")
        
        for var_name, var_info in results.items():
            print_variable(var_name, var_info)
        
        print(f"{'-'*80}\n")
    
    return analysis_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze .pth files from WanVideo data processing pipeline')
    parser.add_argument('pth_file', type=str, help='Path to the .pth file to analyze')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth for nested structure analysis')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    results = analyze_pth_file(args.pth_file, max_depth=args.max_depth, verbose=not args.quiet)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
