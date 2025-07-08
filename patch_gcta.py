#!/usr/bin/env python3
"""
Precise patcher for gcta_nr_robust to fix the time() vsyscall issue
"""
import struct
import sys
import shutil
import os

def patch_time_function(filename):
    """Patch the time() function to use a syscall instead of vsyscall"""
    
    print(f"Patching {filename}...")
    
    # Make backup
    backup_name = filename + '.backup'
    if not os.path.exists(backup_name):
        shutil.copy2(filename, backup_name)
        print(f"Backup created: {backup_name}")
    
    with open(filename, 'rb') as f:
        data = bytearray(f.read())
    
    # The time() function starts at 0xb407e0
    # File offset = address - base address (0x400000)
    time_func_offset = 0xb407e0 - 0x400000
    
    print(f"Patching time() function at offset 0x{time_func_offset:x}")
    
    # The original function is:
    # b407e0:  48 83 ec 08             sub    $0x8,%rsp
    # b407e4:  48 c7 c0 00 04 60 ff    mov    $0xffffffffff600400,%rax
    # b407eb:  ff d0                   call   *%rax
    # b407ed:  48 83 c4 08             add    $0x8,%rsp
    # b407f1:  c3                      ret
    
    # We'll replace it with a direct syscall:
    # mov rax, 201  (syscall number for time on x86_64)
    # xor rdi, rdi  (NULL argument)
    # syscall
    # ret
    # nop padding
    
    new_time_function = [
        0x48, 0xc7, 0xc0, 0xc9, 0x00, 0x00, 0x00,  # mov rax, 201 (sys_time)
        0x48, 0x31, 0xff,                            # xor rdi, rdi
        0x0f, 0x05,                                  # syscall
        0xc3,                                        # ret
        0x90, 0x90, 0x90, 0x90, 0x90,              # nop padding
        0x90, 0x90, 0x90, 0x90, 0x90,
        0x90, 0x90, 0x90, 0x90, 0x90,
        0x90, 0x90
    ]
    
    # Verify we're patching the right location
    expected_start = [0x48, 0x83, 0xec, 0x08]  # sub $0x8,%rsp
    actual_start = data[time_func_offset:time_func_offset+4]
    
    if list(actual_start) == expected_start:
        print("Found expected time() function signature")
        
        # Apply the patch
        for i, byte in enumerate(new_time_function):
            if time_func_offset + i < len(data):
                data[time_func_offset + i] = byte
        
        # Write the patched file
        output_name = filename + '.patched'
        with open(output_name, 'wb') as f:
            f.write(data)
        
        # Make executable
        os.chmod(output_name, 0o755)
        
        print(f"Patch applied successfully!")
        print(f"Output written to: {output_name}")
        print(f"\nTry running:")
        print(f"  ./{output_name} --help")
        
        return True
    else:
        print(f"ERROR: Unexpected bytes at time() function offset")
        print(f"Expected: {expected_start}")
        print(f"Found: {list(actual_start)}")
        return False

def create_wrapper_script(binary_name):
    """Create a wrapper script that uses syscall override"""
    wrapper_name = f"{binary_name}_wrapper.sh"
    
    with open(wrapper_name, 'w') as f:
        f.write(f"""#!/bin/bash
# Wrapper script for {binary_name}

# Create a small library that overrides time() with syscall
cat > /tmp/time_syscall.c << 'EOF'
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

time_t time(time_t *tloc) {{
    return syscall(SYS_time, tloc);
}}
EOF

# Compile it
gcc -shared -fPIC -o /tmp/time_syscall.so /tmp/time_syscall.c 2>/dev/null

# Run with the override (won't work for static binaries, but worth trying)
LD_PRELOAD=/tmp/time_syscall.so ./{binary_name} "$@"
""")
    
    os.chmod(wrapper_name, 0o755)
    print(f"Also created wrapper script: {wrapper_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 precise_patch_gcta.py gcta_nr_robust")
        sys.exit(1)
    
    if patch_time_function(sys.argv[1]):
        create_wrapper_script(sys.argv[1])
        print("\nIf the patched binary doesn't work, try running it in a container:")
        print("  docker run --rm -it -v $(pwd):/work centos:5 /work/gcta_nr_robust.patched --help")
    else:
        print("Patching failed!")