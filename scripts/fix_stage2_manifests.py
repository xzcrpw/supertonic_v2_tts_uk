#!/usr/bin/env python3
"""
Fix Stage 2 manifests - keep ONLY Ukrainian (OpenTTS) data.
LibriTTS with <unk> tokens is poisoning the training.
"""

import json
import os
from pathlib import Path

def fix_manifests():
    manifests_dir = Path("data/manifests_stage2")
    
    for manifest_name in ["train.json", "val.json"]:
        manifest_path = manifests_dir / manifest_name
        backup_path = manifests_dir / f"{manifest_name}.backup_with_libritts"
        
        print(f"\n{'='*60}")
        print(f"Processing {manifest_name}")
        print(f"{'='*60}")
        
        # Read manifest
        with open(manifest_path) as f:
            content = f.read()
        
        # Parse
        try:
            samples = json.loads(content)
        except json.JSONDecodeError:
            samples = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except:
                        continue
        
        print(f"Total samples: {len(samples)}")
        
        # Count by source
        sources = {}
        for s in samples:
            path = s.get('audio_path', '')
            if 'opentts' in path:
                src = 'opentts'
            elif 'libritts' in path:
                src = 'libritts'
            elif 'vctk' in path:
                src = 'vctk'
            else:
                src = 'unknown'
            sources[src] = sources.get(src, 0) + 1
        
        print(f"Sources: {sources}")
        
        # Backup original
        os.rename(manifest_path, backup_path)
        print(f"Backed up to {backup_path}")
        
        # Filter to OpenTTS only
        opentts_samples = [s for s in samples if 'opentts' in s.get('audio_path', '')]
        
        # Also filter for samples with text
        opentts_with_text = [s for s in opentts_samples if len(s.get('text', '')) > 5]
        
        print(f"OpenTTS samples: {len(opentts_samples)}")
        print(f"OpenTTS with text (>5 chars): {len(opentts_with_text)}")
        
        # Write new manifest (JSON Lines format)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for s in opentts_with_text:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(opentts_with_text)} samples to {manifest_path}")
    
    print("\n" + "="*60)
    print("DONE! Now restart Stage 2 training FROM SCRATCH:")
    print("="*60)
    print("""
1. Kill current training:
   pkill -f train_text_to_latent

2. Remove old checkpoints:
   rm -rf outputs/text_to_latent/

3. Restart training:
   torchrun --nproc_per_node=4 train_text_to_latent.py \\
       --config config/22khz_optimal.yaml \\
       --autoencoder_checkpoint checkpoints/autoencoder/checkpoint_150000.pt
""")

if __name__ == '__main__':
    fix_manifests()
