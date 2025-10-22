#!/usr/bin/env python3
"""
Mine GitHub repositories by cloning them locally.

This script reads a list of GitHub repository URLs from a file,
clones them with depth=1 into a specified output directory,
and uses multi-threading for parallel downloads.
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import subprocess
from tqdm import tqdm


def parse_repo_url(url):
    """
    Extract owner and repo name from GitHub URL.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Tuple of (owner, repo_name) or None if invalid
    """
    url = url.strip()
    if not url:
        return None
    
    # Handle both https and git URLs
    if url.endswith('.git'):
        url = url[:-4]
    
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) >= 2:
        owner = path_parts[0]
        repo = path_parts[1]
        return owner, repo
    
    return None


def clone_repo(url, output_dir, max_retries=2):
    """
    Clone a single repository with depth=1.
    
    Args:
        url: Repository URL
        output_dir: Base directory for cloned repos
        max_retries: Number of retry attempts
        
    Returns:
        Tuple of (success: bool, repo_path: str, message: str)
    """
    parsed = parse_repo_url(url)
    if not parsed:
        return False, None, f"Invalid URL: {url}"
    
    owner, repo = parsed
    repo_path = os.path.join(output_dir, owner, repo)
    
    # Skip if already exists
    if os.path.exists(repo_path):
        return True, repo_path, f"Already exists: {owner}/{repo}"
    
    # Create parent directory
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)
    
    # Clone with depth=1
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                ['git', 'clone', '--depth=1', url, repo_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, repo_path, f"Cloned: {owner}/{repo}"
            else:
                error_msg = result.stderr.strip()
                if attempt < max_retries:
                    continue
                return False, None, f"Failed {owner}/{repo}: {error_msg}"
                
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                continue
            return False, None, f"Timeout cloning {owner}/{repo}"
        except Exception as e:
            if attempt < max_retries:
                continue
            return False, None, f"Error cloning {owner}/{repo}: {str(e)}"
    
    return False, None, f"Failed after {max_retries} retries: {owner}/{repo}"


def mine_github_repos(repos_file, output_dir, max_workers=8):
    """
    Clone multiple GitHub repositories in parallel.
    
    Args:
        repos_file: Path to file containing repository URLs (one per line)
        output_dir: Directory to clone repositories into
        max_workers: Number of parallel download threads
        
    Returns:
        Tuple of (success_count, total_count, cloned_paths)
    """
    # Read repository URLs
    with open(repos_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if not urls:
        print("No repository URLs found in file.")
        return 0, 0, []
    
    print(f"Found {len(urls)} repositories to clone.")
    print(f"Output directory: {output_dir}")
    print(f"Using {max_workers} worker threads.\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone repositories in parallel
    success_count = 0
    cloned_paths = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(clone_repo, url, output_dir): url
            for url in urls
        }
        
        # Process results with progress bar
        with tqdm(total=len(urls), desc="Cloning repos") as pbar:
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success, repo_path, message = future.result()
                    if success:
                        success_count += 1
                        if repo_path:
                            cloned_paths.append(repo_path)
                    
                    # Print result
                    status = "✓" if success else "✗"
                    tqdm.write(f"{status} {message}")
                    
                except Exception as e:
                    tqdm.write(f"✗ Unexpected error for {url}: {str(e)}")
                
                pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Cloning complete: {success_count}/{len(urls)} successful")
    print(f"{'='*60}")
    
    return success_count, len(urls), cloned_paths


def main():
    parser = argparse.ArgumentParser(
        description="Mine GitHub repositories by cloning them locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python mine_github.py --repos-file repos.txt --out-dir data/raw_repos --max-workers 8
        """
    )
    
    parser.add_argument(
        '--repos-file',
        type=str,
        required=True,
        help='Path to file containing GitHub repository URLs (one per line)'
    )
    
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/raw_repos',
        help='Output directory for cloned repositories (default: data/raw_repos)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Number of parallel download threads (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.repos_file):
        print(f"Error: Repository file not found: {args.repos_file}")
        sys.exit(1)
    
    # Run mining
    success_count, total_count, cloned_paths = mine_github_repos(
        args.repos_file,
        args.out_dir,
        args.max_workers
    )
    
    # Exit with appropriate code
    if success_count == 0:
        print("\nError: No repositories were successfully cloned.")
        sys.exit(1)
    elif success_count < total_count:
        print(f"\nWarning: {total_count - success_count} repositories failed to clone.")
        sys.exit(0)
    else:
        print("\nAll repositories cloned successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
