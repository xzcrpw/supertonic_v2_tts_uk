#!/usr/bin/env python3
"""Debug script to compare GT vs Generated latents from Stage 2 TTS."""

import torch
import torchaudio
import json
import sys
import os
import glob

def find_tts_checkpoint():
    """Find the latest TTS checkpoint in various possible locations."""
    search_paths = [
        'outputs/text_to_latent/checkpoints/*.pt',  # Found here!
        'checkpoints/tts/*.pt',
        'checkpoints/text_to_latent/*.pt',
        'checkpoints/stage2/*.pt',
        'outputs/*.pt',
        'outputs/tts/*.pt',
        'outputs/checkpoints/*.pt',
        '*.pt',
    ]
    
    all_checkpoints = []
    for pattern in search_paths:
        all_checkpoints.extend(glob.glob(pattern))
    
    # Filter for TTS-related checkpoints
    tts_ckpts = [c for c in all_checkpoints if 'autoencoder' not in c.lower()]
    
    if not tts_ckpts:
        # List all directories to help find
        print("No TTS checkpoints found. Searching all directories...")
        for root, dirs, files in os.walk('.'):
            pt_files = [f for f in files if f.endswith('.pt')]
            if pt_files and 'autoencoder' not in root:
                print(f"  {root}: {pt_files[:5]}...")
        return None
    
    # Sort by modification time, get latest
    tts_ckpts.sort(key=os.path.getmtime, reverse=True)
    return tts_ckpts[0]

def main():
    device = 'cuda'
    
    # ========== 1. Check checkpoint structure ==========
    print("=" * 60)
    print("STEP 1: Finding and checking checkpoints")
    print("=" * 60)
    
    enc_ckpt = torch.load('checkpoints/autoencoder/checkpoint_150000.pt', map_location=device)
    print(f"Autoencoder checkpoint keys: {list(enc_ckpt.keys())}")
    
    # Find TTS checkpoint
    tts_path = find_tts_checkpoint()
    if tts_path is None:
        print("\nERROR: Cannot find TTS checkpoint!")
        print("Run: find . -name '*.pt' -type f | head -20")
        sys.exit(1)
    
    print(f"\nFound TTS checkpoint: {tts_path}")
    tts_ckpt = torch.load(tts_path, map_location=device)
    print(f"TTS checkpoint keys: {list(tts_ckpt.keys())}")
    
    # Determine the correct key for model weights
    if 'model_state_dict' in enc_ckpt:
        enc_state = enc_ckpt['model_state_dict']
    elif 'state_dict' in enc_ckpt:
        enc_state = enc_ckpt['state_dict']
    else:
        # Maybe the checkpoint IS the state dict
        enc_state = enc_ckpt
    
    if 'model_state_dict' in tts_ckpt:
        tts_state = tts_ckpt['model_state_dict']
    elif 'state_dict' in tts_ckpt:
        tts_state = tts_ckpt['state_dict']
    else:
        tts_state = tts_ckpt
    
    print(f"\nAutoencoder state dict - first 5 keys: {list(enc_state.keys())[:5]}")
    print(f"TTS state dict - first 5 keys: {list(tts_state.keys())[:5]}")
    
    # ========== 2. Load models ==========
    print("\n" + "=" * 60)
    print("STEP 2: Loading models")
    print("=" * 60)
    
    from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
    from supertonic.models.text_to_latent import TextToLatent
    from supertonic.data.tokenizer import CharacterTokenizer
    
    # Encoder
    enc = LatentEncoder(
        input_dim=100, hidden_dim=512, output_dim=24,
        num_blocks=10, kernel_size=7, intermediate_mult=4
    )
    enc_keys = {k.replace('encoder.', ''): v for k, v in enc_state.items() if k.startswith('encoder.')}
    if not enc_keys:
        # Maybe no prefix
        enc_keys = enc_state
    enc.load_state_dict(enc_keys)
    enc.to(device).eval()
    print("✓ Encoder loaded")
    
    # Decoder
    dec = LatentDecoder(
        input_dim=24, hidden_dim=512, num_blocks=10, kernel_size=7,
        intermediate_mult=4, dilations=[1,2,4,1,2,4,1,1,1,1],
        n_fft=1024, hop_length=256, causal=True,
        use_hifigan=True, upsample_rates=[8,8,2,2],
        upsample_kernel_sizes=[16,16,4,4], upsample_initial_channel=512,
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]
    )
    dec_keys = {k.replace('decoder.', ''): v for k, v in enc_state.items() if k.startswith('decoder.')}
    if not dec_keys:
        dec_keys = enc_state
    dec.load_state_dict(dec_keys)
    dec.to(device).eval()
    print("✓ Decoder loaded")
    
    # TTS - get vocab size from checkpoint
    vocab_key = 'text_encoder.token_embedding.weight'
    if vocab_key in tts_state:
        vocab_size = tts_state[vocab_key].shape[0]
    else:
        vocab_size = 104  # fallback
    print(f"Vocab size from checkpoint: {vocab_size}")
    
    tts = TextToLatent(
        latent_dim=144,
        vocab_size=vocab_size,
        text_embed_dim=128,
        text_hidden_dim=512,
        ref_hidden_dim=128,
        vf_hidden_dim=512,
        num_ref_vectors=50,
        sigma_min=1e-8,
        p_uncond=0.05,
        cfg_scale=3.0,
        gamma=10.0
    )
    tts.load_state_dict(tts_state)
    tts.to(device).eval()
    print("✓ TTS loaded")
    
    # Tokenizer
    tokenizer = CharacterTokenizer()
    print(f"✓ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    
    # ========== 3. Load test sample ==========
    print("\n" + "=" * 60)
    print("STEP 3: Loading test sample")
    print("=" * 60)
    
    with open('data/manifests_stage2/val.json') as f:
        samples = [json.loads(l) for l in f]
    
    # Find sample with text
    sample = None
    for s in samples:
        text = s.get('text', '')
        if len(text) > 10:
            sample = s
            if 'opentts' in s.get('audio_path', ''):
                break  # Prefer OpenTTS
    
    if sample is None:
        print("ERROR: No sample with text found!")
        sys.exit(1)
    
    print(f"Audio: {sample['audio_path']}")
    print(f"Text: {sample['text']}")
    
    # ========== 4. Process audio ==========
    print("\n" + "=" * 60)
    print("STEP 4: Processing audio")
    print("=" * 60)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, hop_length=256,
        win_length=1024, n_mels=100, f_min=20.0, f_max=11025.0, power=1.0
    ).to(device)
    
    audio, sr = torchaudio.load(sample['audio_path'])
    if sr != 22050:
        audio = torchaudio.functional.resample(audio, sr, 22050)
    audio = audio.mean(0, keepdim=True).to(device)  # [1, samples]
    print(f"Audio shape: {audio.shape}, duration: {audio.shape[1]/22050:.2f}s")
    
    with torch.no_grad():
        mel = mel_transform(audio)  # [1, n_mels, T]
        print(f"Mel shape: {mel.shape}")
        
        # Encode to latent
        latent = enc(mel)  # [1, 24, T]
        print(f"Latent shape: {latent.shape}")
        
        # Compress 6x
        T = latent.shape[-1]
        T_pad = (6 - T % 6) % 6
        if T_pad > 0:
            latent_pad = torch.nn.functional.pad(latent, (0, T_pad))
        else:
            latent_pad = latent
        compressed = latent_pad.reshape(1, 24, -1, 6).permute(0, 1, 3, 2).reshape(1, 144, -1)
        print(f"Compressed GT shape: {compressed.shape}")
        
        # ========== 5. Compare latents ==========
        print("\n" + "=" * 60)
        print("STEP 5: Comparing GT vs Generated latents")
        print("=" * 60)
        
        print(f"GT compressed stats:")
        print(f"  mean: {compressed.mean():.4f}")
        print(f"  std:  {compressed.std():.4f}")
        print(f"  min:  {compressed.min():.4f}")
        print(f"  max:  {compressed.max():.4f}")
        
        # Tokenize
        tokens = tokenizer.encode(sample['text'])
        tokens_t = torch.tensor([tokens], device=device)
        print(f"\nTokens shape: {tokens_t.shape}")
        
        # Generate
        gen_compressed = tts.inference(tokens_t, compressed, num_steps=32)
        print(f"Generated shape: {gen_compressed.shape}")
        
        print(f"\nGenerated compressed stats:")
        print(f"  mean: {gen_compressed.mean():.4f}")
        print(f"  std:  {gen_compressed.std():.4f}")
        print(f"  min:  {gen_compressed.min():.4f}")
        print(f"  max:  {gen_compressed.max():.4f}")
        
        # Difference
        diff = (compressed - gen_compressed).abs().mean()
        print(f"\n>>> L1 difference: {diff:.4f}")
        
        # Correlation
        gt_flat = compressed.flatten()
        gen_flat = gen_compressed.flatten()
        corr = torch.corrcoef(torch.stack([gt_flat, gen_flat]))[0, 1]
        print(f">>> Correlation: {corr:.4f}")
        
        # ========== 6. Decode and save ==========
        print("\n" + "=" * 60)
        print("STEP 6: Decoding and saving audio")
        print("=" * 60)
        
        # Decode GT
        gt_audio = dec(latent)
        torchaudio.save('debug_gt.wav', gt_audio.cpu(), 22050)
        print("✓ Saved debug_gt.wav (ground truth reconstruction)")
        
        # Decode generated
        T_gen = gen_compressed.shape[-1]
        gen_latent = gen_compressed.reshape(1, 24, 6, T_gen).permute(0, 1, 3, 2).reshape(1, 24, T_gen * 6)
        gen_audio = dec(gen_latent)
        torchaudio.save('debug_gen.wav', gen_audio.cpu(), 22050)
        print("✓ Saved debug_gen.wav (TTS generated)")
        
        print("\n" + "=" * 60)
        print("DONE! Compare the two files:")
        print("  - debug_gt.wav  = original audio through autoencoder")
        print("  - debug_gen.wav = TTS generated from text + reference")
        print("=" * 60)
        
        # Interpretation
        print("\n>>> INTERPRETATION:")
        if diff < 0.5 and corr > 0.8:
            print("Model is learning well! Output should be similar to GT.")
        elif diff < 1.0 and corr > 0.5:
            print("Model is learning, but needs more training.")
        elif corr > 0.3:
            print("Model is starting to learn, continue training.")
        else:
            print("Model output is random - check training loop!")

if __name__ == '__main__':
    main()
