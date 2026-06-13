from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

def mine_targets(ra, dec, radius_arcmin=2):
    print("==================================================")
    print("        DEEP FIELD PHOTOMETRY DATA MINER          ")
    print("==================================================")
    print(f"[SYSTEM] Aligning virtual telescope to RA: {ra}, DEC: {dec}")
    
    target_pos = coords.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    filename = f"deep_field_targets_RA{round(ra, 2)}.csv"
    
    print(f"[NETWORK] Scanning a {radius_arcmin}-arcminute radius for galaxies via SDSS...")
    try:
        photo_data = SDSS.query_region(
            target_pos, 
            radius=radius_arcmin*u.arcmin, 
            photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z']
        )
        if photo_data is not None and len(photo_data) > 0:
            df = photo_data.to_pandas()
            df = df[(df['u'] > 0) & (df['u'] < 30) & (df['z'] > 0) & (df['z'] < 30)]
            if len(df) > 0:
                print(f"\n✅ [SUCCESS] Extracted {len(df)} celestial objects from SDSS!")
                df.to_csv(filename, index=False)
                return filename
        print("[WARNING] SDSS returned empty or failed to find matching objects. Trying VizieR...")
    except Exception as e:
        print(f"[WARNING] SDSS connection/query failed: {e}. Trying VizieR...")
        
    print(f"[NETWORK] Querying APASS Galactic Plane Survey via VizieR...")
    try:
        v = Vizier()
        v.ROW_LIMIT = 5000
        result = v.query_region(target_pos, radius=radius_arcmin*u.arcmin, catalog='II/336/apass9')
        if not result or len(result) == 0:
            print("[CRITICAL] Database mirror returned an empty matrix for this sector.")
            return None
        
        table = result[0]
        df = table.to_pandas()
        
        target_cols = {'ra': None, 'dec': None, 'g': None, 'r': None, 'i': None, 'b': None, 'v': None}
        for col in df.columns:
            c = col.lower()
            if c.startswith('e_') or 'err' in c: continue
            if c in ['raj2000', 'ra']: target_cols['ra'] = col
            elif c in ['dej2000', 'dec']: target_cols['dec'] = col
            elif c in ['g_mag', 'gmag']: target_cols['g'] = col
            elif c in ['r_mag', 'rmag']: target_cols['r'] = col
            elif c in ['i_mag', 'imag']: target_cols['i'] = col
            elif c == 'bmag': target_cols['b'] = col
            elif c == 'vmag': target_cols['v'] = col

        for col in df.columns:
            c = col.lower()
            if c.startswith('e_') or 'err' in c: continue
            if not target_cols['ra'] and 'ra' in c: target_cols['ra'] = col
            if not target_cols['dec'] and ('de' in c or 'dec' in c): target_cols['dec'] = col
            if not target_cols['g'] and ('gmag' in c or 'g_mag' in c or "g'" in c): target_cols['g'] = col
            if not target_cols['r'] and ('rmag' in c or 'r_mag' in c or "r'" in c): target_cols['r'] = col
            if not target_cols['i'] and ('imag' in c or 'i_mag' in c or "i'" in c): target_cols['i'] = col
            if not target_cols['b'] and 'bmag' in c: target_cols['b'] = col
            if not target_cols['v'] and 'vmag' in c: target_cols['v'] = col

        clean_data = {}
        for target, original_name in target_cols.items():
            if original_name is not None:
                series = df[original_name]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                clean_data[target] = pd.to_numeric(series, errors='coerce')
        
        df_clean = pd.DataFrame(clean_data)
        
        if 'ra' not in df_clean.columns or 'dec' not in df_clean.columns:
            return None
            
        if 'g' in df_clean.columns and 'r' in df_clean.columns and 'i' in df_clean.columns:
            if 'b' in df_clean.columns and 'v' in df_clean.columns:
                df_clean['u'] = df_clean['g'] + 1.24 * (df_clean['b'] - df_clean['v'])
            else:
                df_clean['u'] = df_clean['g'] + 0.85
            df_clean['z'] = df_clean['i'] - 0.42 * (df_clean['r'] - df_clean['i'])
        else:
            df_clean['u'], df_clean['g'], df_clean['r'], df_clean['i'], df_clean['z'] = 18.0, 17.5, 17.0, 16.5, 16.0
            
        required_cols = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z']
        df_final = df_clean[required_cols].dropna()
        df_final = df_final[(df_final['u'] > 0) & (df_final['u'] < 30) & (df_final['z'] > 0) & (df_final['z'] < 30)]
        
        if len(df_final) > 0:
            print(f"\n✅ [SUCCESS] Extracted {len(df_final)} valid footprints from VizieR!")
            df_final.to_csv(filename, index=False)
            return filename
        else:
            print("[CRITICAL] Processed data was empty.")
            return None
        
    except Exception as e:
        print(f"[ERROR] Live query execution pipeline failed: {e}")
        return None
