#!/usr/bin/env bash
# Post-provisioning script to add custom startup scripts
# Run this after provisioning to add the phone-only startup script

WORKSPACE_DIR="/tmp/nvflare/workspaces/edge_example/prod_00"
TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Adding custom phone-only startup script..."

# Copy the start_phone.sh template to the workspace
if [ -f "$TEMPLATE_DIR/start_phone.sh.template" ]; then
    cp "$TEMPLATE_DIR/start_phone.sh.template" "$WORKSPACE_DIR/start_phone.sh"
    chmod +x "$WORKSPACE_DIR/start_phone.sh"
    echo "✓ Created start_phone.sh in workspace"
else
    echo "⚠ Warning: start_phone.sh.template not found"
fi

echo "Patching configurations for insecure (HTTP) mode..."

# Function to patch fed_server.json
find "$WORKSPACE_DIR" -name "fed_server.json" | while read file; do
    # Replace https with http and disable secure connection
    sed -i '' 's/"scheme": "https"/"scheme": "http"/g' "$file"
    echo "✓ Patched $file to HTTP"
done

# Function to patch fed_relay.json (for R1/R2 relays)
find "$WORKSPACE_DIR" -name "fed_relay.json" | while read file; do
    sed -i '' 's/"scheme": "https"/"scheme": "http"/g' "$file"
    
    # Patch the relay connect_to section - connection_security must be "clear" not empty!
    python3 -c "
import json
try:
    with open('$file', 'r') as f:
        data = json.load(f)
    
    if 'connect_to' in data:
        ct = data['connect_to']
        ct['scheme'] = 'http'
        ct['connection_security'] = 'clear'  # Must be 'clear' not empty!
        
    with open('$file', 'w') as f:
        json.dump(data, f, indent=2)
    print('✓ Patched relay JSON in $file')
except Exception as e:
    print(f'Error patching $file: {e}')
"
done

# Function to patch fed_client.json
find "$WORKSPACE_DIR" -name "fed_client.json" | while read file; do
    # Change scheme to http
    sed -i '' 's/"scheme": "https"/"scheme": "http"/g' "$file"
    
    # Remove SSL keys and change connection_security to empty/none
    # We use a temporary python script for reliable JSON editing
    python3 -c "
import json
import sys
try:
    with open('$file', 'r') as f:
        data = json.load(f)
    
    if 'client' in data:
        cl = data['client']
        if 'ssl_private_key' in cl: del cl['ssl_private_key']
        if 'ssl_cert' in cl: del cl['ssl_cert']
        if 'ssl_root_cert' in cl: del cl['ssl_root_cert']
        cl['connection_security'] = 'clear'  # Must be 'clear' not empty!
        
    with open('$file', 'w') as f:
        json.dump(data, f, indent=2)
    print('✓ Patched JSON in $file')
except Exception as e:
    print(f'Error patching $file: {e}')
"
done

echo "Post-provisioning complete! SSL disabled."

# Patch all sub_start.sh scripts to disable secure_train
echo "Patching startup scripts to disable secure_train..."
find "$WORKSPACE_DIR" -name "sub_start.sh" | while read file; do
    sed -i '' 's/secure_train=true/secure_train=false/g' "$file"
    echo "✓ Patched $file"
done

echo ""
echo "✅ All configurations patched for insecure HTTP mode!"
