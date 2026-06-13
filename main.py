import sys
import os
import scripts.data_miner as data_miner

# Ensure scripts directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import scripts.process_deep_field as process_deep_field
import scripts.pygame_3d_observatory as pygame_3d_observatory

# Map each local image to its real-world space coordinates.
# To find the RA and Dec of any NASA image:
# 1. Look up the image on Wikipedia or the official NASA/Hubble/JWST site.
# 2. Look for 'Coordinates', 'Right Ascension (RA)', and 'Declination (Dec)'.
# 3. If it's in Hours/Minutes/Seconds (e.g., 10h 3m 2s), use an online 'RA/Dec to Decimal Degrees converter'.
# 4. Plug those decimal degrees in here!

SPACE_IMAGES = [
    {
        "image_file": "image.png",
        "name": "Swan Nebula (Messier 17 - NASA SOFIA)",
        "ra": 275.11,
        "dec": -16.18
    },
    {
        "image_file": "image copy.png",
        "name": "Hubble Ultra Deep Field",
        "ra": 53.16,
        "dec": -27.79
    },
    {
        "image_file": "image copy 2.png",
        "name": "Keyhole Nebula Region",
        "ra": 161.26,
        "dec": -59.68
    },
    {
        "image_file": "image copy 3.png",
        "name": "SDSS Northern Deep Field",
        "ra": 184.20,
        "dec": 34.10
    }
]

def run_pipeline():
    print("==================================================")
    print("       STARTING AI COSMIC OBSERVATORY BATCH FLOW  ")
    print("==================================================")
    print("\n[INFO] We will loop through your space images, fetch telemetry,")
    print("predict redshifts with AI, and launch the 3D map for each one.")
    print("[NOTE] Close the Matplotlib Graph and the 3D Window to advance to the next image.\n")
    
    for target in SPACE_IMAGES:
        print("\n" + "="*50)
        print(f"⭐ TARGET: {target['name']} (File: {target['image_file']})")
        print(f"📍 COORDINATES: RA {target['ra']}°, DEC {target['dec']}°")
        print("="*50)

        # Step 1: Mine the photometric data for these coordinates
        print("\n[STEP 1] Mining Photometric Data...")
        csv_file = data_miner.mine_targets(target['ra'], target['dec'])
        
        if not csv_file:
            print("[ERROR] Mining failed for this target. Skipping to next...\")")
            continue

        # Step 2: Process the deep field data to get redshift predictions
        print("\n[STEP 2] Processing Deep Field Data with AI Model...")
        process_deep_field.generate_cosmic_depth_map(csv_file)
        
        mapped_csv = "mapped_" + csv_file
        
        # Step 3: Boot up the 3D interactive visualization
        print("\n[STEP 3] Launching 3D Observatory...")
        pygame_3d_observatory.main(mapped_csv)

    print("\n[SYSTEM] All targets processed. Observatory pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
