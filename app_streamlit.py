import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pickle
import os
import math
from scipy.signal import convolve2d
import scipy.ndimage as ndimage
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="VeinAuth - Dorsal Hand Vein Authentication",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Session State Initialization ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = "Login"
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"
if 'nav_radio' not in st.session_state:
    st.session_state['nav_radio'] = "Home"
if 'reg_success_shown' not in st.session_state:
    st.session_state['reg_success_shown'] = False
if 'reg_seed' not in st.session_state:
    st.session_state['reg_seed'] = 0
if 'admin_authenticated' not in st.session_state:
    st.session_state['admin_authenticated'] = False
if 'admin_reg_seed' not in st.session_state:
    st.session_state['admin_reg_seed'] = 100 # Separate seed for admin registrations

# --- Biometric Data Helper ---
BIO_DB_PATH = os.path.join("models", "biometric_db.pkl")
LOG_FILE_PATH = os.path.join("logs", "login_history.txt")

def add_log(username):
    os.makedirs("logs", exist_ok=True)
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Store exactly: Time,Username
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"{now},{username}\n")

def get_logs():
    if os.path.exists(LOG_FILE_PATH):
        import pandas as pd
        try:
            #Re-read with coloumns for Time and User
            df = pd.read_csv(LOG_FILE_PATH, names=["Timestamp", "Username"])
            # Return only the Timestamp column as requested
            return df
        except Exception:
            return None
    return None

def get_bio_db():
    if os.path.exists(BIO_DB_PATH):
        with open(BIO_DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_bio_db(db):
    os.makedirs("models", exist_ok=True)
    with open(BIO_DB_PATH, 'wb') as f:
        pickle.dump(db, f)

def extract_encoded_features(img_pil, cnn, scaler):
    # Standardize preprocessing
    img_np = np.array(img_pil.resize((224, 224)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = vein_pattern_extraction(gray)
    final_input = np.stack((mask * 255,) * 3, axis=-1).astype(np.uint8)
    
    # AI Encoding
    input_tensor = np.expand_dims(final_input, axis=0).astype(np.float32)
    input_tensor = tf.keras.applications.vgg16.preprocess_input(input_tensor)
    raw_features = cnn.predict(input_tensor, verbose=0)
    scaled_features = scaler.transform(raw_features)
    return scaled_features

def is_hand_image(img_pil, hand_model):
    # Standard preprocessing for hand detection
    img_resized = img_pil.resize((224, 224))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = hand_model.predict(img_array, verbose=0)
    raw_score = float(prediction[0][0])
    
    # Based on training: 0.0 = Hand, 1.0 = Non-Hand (Alphabetical order: Hand, Non hand)
    is_hand = raw_score < 0.5
    hand_confidence = (1.0 - raw_score) # Calculate score where 1.0 is definitely a hand
    
    return is_hand, hand_confidence

# --- Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #fdfbf7;
        color: #1f2937;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fdfbf7 0%, #f5f3ff 100%);
    }

    /* Glassmorphism Card (Premium Pearl) */
    .stCard {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(124, 58, 237, 0.1);
        padding: 3rem;
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.05);
        margin-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }

    .sub-header {
        color: #7c3aed;
        font-weight: 600;
        margin-top: 2rem;
        border-left: 5px solid #fbbf24;
        padding-left: 15px;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 14px;
        height: 3.5em;
        background: linear-gradient(90deg, #7c3aed, #6d28d9);
        color: white !important;
        font-weight: 700;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 20px rgba(124, 58, 237, 0.25);
        background: linear-gradient(90deg, #8b5cf6, #7c3aed);
    }

    /* Tab Styling (Violet Theme) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: white;
        padding: 8px 15px;
        border-radius: 100px;
        border: 1px solid rgba(124, 58, 237, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent !important;
        border-radius: 100px;
        border: none;
        color: #6b7280 !important;
        font-weight: 600 !important;
        padding: 0 25px !important;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }

    /* Notification Styling */
    div[data-testid="stNotification"] {
        border-radius: 16px !important;
        border-left: 6px solid #fbbf24 !important;
        background-color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }

    /* Navigation Header Info */
    .user-info {
        text-align: right; 
        color: #7c3aed; 
        font-size: 0.95rem; 
        font-weight: 600;
        margin-top: 10px;
    }

    /* Hide sidebar and streamline top */
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Image Processing Functions (Ported Logic) ---

def normalize_data(x, low=0, high=1, data_type=None):
    x = np.asarray(x, dtype=np.float64)
    min_x, max_x = np.min(x), np.max(x)
    if max_x - min_x == 0: return x
    x = (x - float(min_x)) / float((max_x - min_x))
    x = x * (high - low) + low
    return np.asarray(x, dtype=data_type if data_type else np.float64)

def remove_hair(image, kernel_size):
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    return convolve2d(image, kernel, mode='same', fillvalue=0)

def compute_curvature(image, sigma):
    winsize = int(np.ceil(4 * sigma))
    window = np.arange(-winsize, winsize + 1)
    X, Y = np.meshgrid(window, window)
    G = (1.0 / (2 * math.pi * sigma ** 2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
    G1_90, G2_90 = G1_0.T, G2_0.T
    hxy = ((X * Y) / (sigma ** 8)) * G
    i_g1_0, i_g2_0 = 0.1 * ndimage.convolve(image, G1_0), 10 * ndimage.convolve(image, G2_0)
    i_g1_90, i_g2_90 = 0.1 * ndimage.convolve(image, G1_90), 10 * ndimage.convolve(image, G2_90)
    fxy = ndimage.convolve(image, hxy)
    i_g1_45, i_g1_m45 = 0.5*np.sqrt(2)*(i_g1_0+i_g1_90), 0.5*np.sqrt(2)*(i_g1_0-i_g1_90)
    i_g2_45, i_g2_m45 = 0.5*i_g2_0+fxy+0.5*i_g2_90, 0.5*i_g2_0-fxy+0.5*i_g2_90
    return np.dstack([(i_g2_0/((1+i_g1_0**2)**1.5)), (i_g2_90/((1+i_g1_90**2)**1.5)), 
                      (i_g2_45/((1+i_g1_45**2)**1.5)), (i_g2_m45/((1+i_g1_m45**2)**1.5))])

def binaries(G):
    valid = G[G > 0]
    return (G > np.median(valid)).astype(np.float64) if len(valid) > 0 else np.zeros_like(G)

def connect_profile_1d(vp):
    return np.amin([np.amax([vp[3:-1], vp[4:]], axis=0), np.amax([vp[1:-3], vp[:-4]], axis=0)], axis=0)

def connect_centres(vein_score):
    connected_center = np.zeros(vein_score.shape, dtype='float64')
    # Original Miura technique: sum of all directional scores
    vein_score_sum = np.sum(vein_score, axis=2)
    
    # Horizontal direction
    for index in range(vein_score_sum.shape[0]):
        connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score_sum[index, :])

    # Vertical direction
    for index in range(vein_score_sum.shape[1]):
        connected_center[2:-2, index, 1] = connect_profile_1d(vein_score_sum[:, index])

    # Diagonals (Real Miura implementation)
    i, j = np.indices(vein_score_sum.shape)
    border = np.zeros((2,), dtype='float64')
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        connected_center[:, :, 2][i == (j - index)] = np.hstack([border, connect_profile_1d(vein_score_sum.diagonal(index)), border])

    Vud = np.flipud(vein_score_sum)
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        mask = (i == (j - index))
        connected_center[:, :, 3][np.flipud(mask)] = np.hstack([border, connect_profile_1d(Vud.diagonal(index)), border])

    return connected_center

def profile_score_1d(p):
    t = (p > 0).astype(int)
    d = t[1:] - t[:-1]
    starts, ends = np.argwhere(d > 0).flatten() + 1, np.argwhere(d < 0).flatten() + 1
    if t[0]: starts = np.insert(starts, 0, 0)
    if t[-1]: ends = np.append(ends, len(p))
    s = np.zeros_like(p)
    for start, end in zip(starts, ends):
        chunk = p[int(start):int(end)]
        if len(chunk) > 0: s[int(start) + np.argmax(chunk)] = np.max(chunk) * (end - start)
    return s

def compute_vein_score(k):
    score = np.zeros(k.shape, dtype='float64')
    # Horizontal
    for index in range(k.shape[0]):
        score[index, :, 0] += profile_score_1d(k[index, :, 0])
    # Vertical
    for index in range(k.shape[1]):
        score[:, index, 1] += profile_score_1d(k[:, index, 1])
    # 45 degrees
    i, j = np.indices(k.shape[:2])
    for index in range(-k.shape[0] + 1, k.shape[1]):
        score[i == (j - index), 2] += profile_score_1d(k[:, :, 2].diagonal(index))
    # -45 degrees
    curve_m45 = np.flipud(k[:, :, 3])
    score_m45 = np.zeros_like(curve_m45)
    for index in range(-k.shape[0] + 1, k.shape[1]):
        score_m45[i == (j - index)] += profile_score_1d(curve_m45.diagonal(index))
    score[:, :, 3] = np.flipud(score_m45)
    return score

def vein_pattern_extraction(image):
    data = np.asarray(image, dtype=np.float64)
    f = remove_hair(data, 6)
    p = normalize_data(f, 0, 255)
    kappa = compute_curvature(p, sigma=8)
    score = compute_vein_score(kappa)
    conect = connect_centres(score)
    threshold = binaries(np.amax(conect, axis=2))
    return np.multiply(image, threshold, dtype=np.float64), threshold

# --- Model Loading ---

@st.cache_resource
def load_assets():
    cnn = tf.keras.models.load_model(os.path.join("models", "cnn_model.h5"))
    hand_model = tf.keras.models.load_model(os.path.join("models", "hand_detection_model.h5"))
    with open(os.path.join("models", "Svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)
    with open(os.path.join("models", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join("models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return cnn, hand_model, svm, le, scaler

# --- Page Functions ---

def show_home():
    st.markdown('<h1 class="main-header">Dorsal Hand Vein-Based Authentication</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Our system allows users to authenticate their identity using the unique patterns of veins on the dorsal side of their hand. 
        This biometric authentication system utilizes near-infrared imaging technology and advanced image processing techniques to detect vein patterns, 
        which are highly secure and reliable.
        """)
        
        st.markdown('<h3 class="sub-header">Key Features</h3>', unsafe_allow_html=True)
        st.markdown("""
        *   **Biometric Authentication**: Unique vein pattern for identity verification.
        *   **Non-Invasive**: No physical contact required, ensuring comfort.
        *   **High Security**: Vein patterns are virtually impossible to forge.
        *   **Neural Intelligence**: Powered by CNN for feature extraction and SVM for classification.
        """)
        
        st.markdown('<h3 class="sub-header">Instructions</h3>', unsafe_allow_html=True)
        st.info("""
        1. **Login** or **Register** to access the Authentication portal.
        2. **Upload** a clear image of your dorsal hand.
        3. **Wait** while our AI extracts your unique vein features.
        4. **Verify** your identity against our registered database.
        """)
    
    with col2:
        # Placeholder for an image or a decorative element
        img_path = os.path.join("assets", "image.jpg")
        if os.path.exists(img_path):
            try:
                display_img = Image.open(img_path)
                st.image(display_img, width="stretch")
            except Exception:
                pass

def show_about():
    st.markdown('<h1 class="main-header">About the System</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Overview</h3>', unsafe_allow_html=True)
        st.write("""
        Biometric systems have become increasingly important in enhancing security and privacy. Among biometric modalities, 
        dorsal hand vein-based authentication offers a robust and secure solution. This system leverages near-infrared imaging 
        to detect the unique vein patterns beneath the skin, providing protection from spoofing attacks.
        """)
        
        st.markdown('<h3 class="sub-header">Technical Benefits</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Accuracy**: Captures unique internal structures that don't change over time.
        - **Security**: Resistant to daylight spoofing (unlike face) and surface replica (unlike fingerprints).
        - **Hygiene**: Non-contact design makes it perfect for shared environments.
        """)
        
        st.markdown('<h3 class="sub-header">Applications</h3>', unsafe_allow_html=True)
        st.write("- Government facilities & Banking security\n- Restricted workspace access\n- E-commerce identity verification")
        st.markdown('</div>', unsafe_allow_html=True)

def show_auth_system():
    if not st.session_state['authenticated']:
        st.warning("Please Login to access the dashboard.")
        return

    st.markdown(f"""
        <h1 class="main-header">User Dashboard</h1>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color: #1e3a8a;">Welcome, {st.session_state['username']}! 👋</h2>
            <p style="font-size: 1.2rem; color: #475569; margin-top: 20px;">
                You have successfully authenticated using your high-security biometric vein patterns.
            </p>
            <hr style="margin: 30px 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #3b82f6, transparent);">
            <p style="color: #64748b;">
                You are now logged into the secure zone of the <b>VeinAuth</b> system. 
                Your unique biometric identity has been verified against our biometric template database.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="sub-header">Security Status</h3>', unsafe_allow_html=True)
    st.success(f"Verified Profile: **{st.session_state['username']}**")
    st.info("Your session is secured with end-to-end biometric validation.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Navigation Architecture ---

def on_nav_change():
    st.session_state['current_page'] = st.session_state['nav_radio']

def on_auth_mode_change():
    st.session_state['auth_mode'] = st.session_state['auth_radio']

def on_reg_input_change():
    if 'reg_success_msg' in st.session_state:
        del st.session_state['reg_success_msg']

def show_login_register():
    cnn, hand_model, svm, le, scaler = load_assets()
    db = get_bio_db()
    tab_login, tab_reg, tab_admin = st.tabs(["Login", "Register", "Admin Access"])
    
    with tab_login:
        st.markdown('<h1 class="main-header">Biometric Login</h1>', unsafe_allow_html=True)

        st.subheader("Login into your Account")
        user = st.text_input("Enter Registered Name", key="login_user")
        hand_side = st.selectbox("Select Hand for Verification", ["Left Hand", "Right Hand"])
        uploaded_hand = st.file_uploader(f"Upload {hand_side} Image", type=["png", "jpg", "jpeg"], key="login_file")
        
        if st.button("🚀 Verify & Login", key="login_btn"):
            if not user or not uploaded_hand:
                st.error("Please provide both Name and Hand Image.")
            elif user not in db:
                st.error("User not found in database.")
            else:
                side_key = "left" if hand_side == "Left Hand" else "right"
                
                # Check for old vs new database format
                if not isinstance(db[user], dict):
                    st.error("⚠️ This account was registered with an older biometric format. Please contact admin or re-register.")
                elif side_key not in db[user]:
                    st.error(f"No registration found for {hand_side}. Please register it first.")
                else:
                    with st.spinner(f"Verifying {hand_side} Biometrics..."):
                        img_input = Image.open(uploaded_hand).convert('RGB')
                        
                        # --- Hand Detection Validation ---
                        is_hand, conf = is_hand_image(img_input, hand_model)
                        if not is_hand:
                            st.error(f"❌ **Validation Failed:** The uploaded image does not appear to be a hand (Confidence: {conf:.2%}). Please upload a clear photo of your dorsal hand vein pattern.")
                        else:
                            # 1. Extract Current Features
                            current_vector = extract_encoded_features(img_input, cnn, scaler)
                            stored_vector = db[user][side_key]
                            
                            # 2. Compare using Cosine Similarity
                            similarity = cosine_similarity(current_vector, stored_vector)[0][0]
                            
                            if similarity > 0.85: # High threshold
                                st.session_state['authenticated'] = True
                                st.session_state['username'] = user
                                add_log(user) # Log the login
                                st.success(f"Access Granted! Match: {similarity:.2%}")
                                st.rerun()
                            else:
                                st.error(f"Access Denied! Biometric mismatch (Match: {similarity:.2%})")

    with tab_reg:
        st.markdown('<h1 class="main-header">Biometric Registration</h1>', unsafe_allow_html=True)
        
        # Show registration success message here
        if 'reg_success_msg' in st.session_state and st.session_state['reg_success_msg']:
            st.success(st.session_state['reg_success_msg'])
            # We keep it for one render cycle or until cleared
            st.session_state['reg_success_shown'] = True

        st.subheader("Register New Biometric Profile")
        reg_seed = st.session_state['reg_seed']
        new_user = st.text_input("Enter Full Name", key=f"reg_name_{reg_seed}", on_change=on_reg_input_change)
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("### Left Hand")
            reg_hand_l = st.file_uploader("Upload Left Hand Template", type=["png", "jpg", "jpeg"], key=f"reg_l_{reg_seed}", on_change=on_reg_input_change)
        with col_r:
            st.write("### Right Hand")
            reg_hand_r = st.file_uploader("Upload Right Hand Template", type=["png", "jpg", "jpeg"], key=f"reg_r_{reg_seed}", on_change=on_reg_input_change)
        
        st.caption("ℹ️ *For maximum security, ensure your Left and Right hand images are distinct and correctly placed.*")
        
        if st.button("📝 Register Identity", key="reg_btn"):
            if not new_user or not reg_hand_l or not reg_hand_r:
                st.error("Name and BOTH hand images are mandatory.")
            elif new_user.strip().isdigit():
                st.error("⚠️ **Invalid Name:** Username cannot consist only of numbers. Please use a valid name.")
            elif new_user.strip().lower() in [u.lower() for u in db.keys()]:
                st.warning(f"⚠️ The name '{new_user}' is already taken. Please use a unique name.")
            else:
                with st.spinner("Validating and Encoding Biometric Features..."):
                    img_l = Image.open(reg_hand_l).convert('RGB')
                    img_r = Image.open(reg_hand_r).convert('RGB')
                    
                    # --- Hand Detection Validation ---
                    is_l, conf_l = is_hand_image(img_l, hand_model)
                    is_r, conf_r = is_hand_image(img_r, hand_model)
                    
                    if not is_l or not is_r:
                        failed_details = []
                        if not is_l: failed_details.append(f"Left Hand ({conf_l:.1%})")
                        if not is_r: failed_details.append(f"Right Hand ({conf_r:.1%})")
                        st.error(f"❌ **Validation Failed:** Detection system failed for: {', '.join(failed_details)}. Please ensure the images clearly show your hand.")
                    else:
                        features_l = extract_encoded_features(img_l, cnn, scaler)
                        features_r = extract_encoded_features(img_r, cnn, scaler)

                        # --- Self-Deduplication Check (Ensure L and R are different) ---
                        self_sim = cosine_similarity(features_l, features_r)[0][0]
                        if self_sim > 0.90:
                            st.error("⚠️ **Registration Error:** The uploaded Left and Right hand images appear to be identical or too similar. Please ensure you upload the correct unique image for each hand.")
                        else:
                            # --- Biometric De-duplication Check (Against Other Users) ---
                            duplicate_found = False
                            for existing_user, vectors in db.items():
                                if isinstance(vectors, dict) and 'left' in vectors and 'right' in vectors:
                                    sim_ll = cosine_similarity(features_l, vectors['left'])[0][0]
                                    sim_lr = cosine_similarity(features_l, vectors['right'])[0][0]
                                    sim_rl = cosine_similarity(features_r, vectors['left'])[0][0]
                                    sim_rr = cosine_similarity(features_r, vectors['right'])[0][0]
                                    
                                    if any(s > 0.95 for s in [sim_ll, sim_lr, sim_rl, sim_rr]):
                                        duplicate_found = True
                                        st.error(f"⚠️ **Security Alert:** This biometric profile is already registered. Duplicate identities are not permitted.")
                                        break
                        
                        if not duplicate_found:
                            db[new_user] = {
                                "left": features_l,
                                "right": features_r
                            }
                            save_bio_db(db)
                            
                            # Set success message
                            st.session_state['reg_success_msg'] = f"✅ Registration Successful for **{new_user}**! You can now login with your biometrics."
                            
                            # Increment seed to clear all registration widgets at once
                            st.session_state['reg_seed'] += 1
                            
                            st.rerun()

    with tab_admin:
        st.markdown('<h1 class="main-header">Admin Login</h1>', unsafe_allow_html=True)
        st.subheader("Administrative Control Center")
        admin_u = st.text_input("Admin Username", key="admin_u")
        admin_p = st.text_input("Admin Password", type="password", key="admin_p")
        
        if st.button("🔑 Access Admin Panel", key="admin_login_btn"):
            if admin_u == "admin" and admin_p == "admin":
                st.session_state['admin_authenticated'] = True
                st.success("Admin Access Granted!")
                st.rerun()
            else:
                st.error("Invalid Admin Credentials.")

def show_admin_panel():
    st.markdown('<h1 class="main-header">Admin Control Panel</h1>', unsafe_allow_html=True)
    cnn, hand_model, svm, le, scaler = load_assets()
    db = get_bio_db()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Registered Users")
        if not db:
            st.info("No users registered yet.")
        else:
            for user in list(db.keys()):
                c1, c2 = st.columns([3, 1])
                c1.write(f"👤 **{user}**")
                if c2.button("🗑️", key=f"admin_del_{user}"):
                    del db[user]
                    save_bio_db(db)
                    st.success(f"User '{user}' deleted.")
                    st.rerun()

        st.markdown('<hr style="border: 1px solid rgba(124, 58, 237, 0.1);">', unsafe_allow_html=True)
        st.subheader("📝 Register New User as Admin")
        
        a_reg_seed = st.session_state['admin_reg_seed']
        a_new_user = st.text_input("Enter Full Name", key=f"admin_reg_name_{a_reg_seed}")
        a_reg_hand_l = st.file_uploader("Upload Left Hand Template", type=["png", "jpg", "jpeg"], key=f"admin_reg_l_{a_reg_seed}")
        a_reg_hand_r = st.file_uploader("Upload Right Hand Template", type=["png", "jpg", "jpeg"], key=f"admin_reg_r_{a_reg_seed}")
        
        if st.button("📝 Register Identity", key="admin_reg_btn"):
            if not a_new_user or not a_reg_hand_l or not a_reg_hand_r:
                st.error("Name and BOTH hand images are mandatory.")
            elif a_new_user.strip().isdigit():
                st.error("⚠️ **Invalid Name:** Username cannot consist only of numbers.")
            elif a_new_user.strip().lower() in [u.lower() for u in db.keys()]:
                st.warning(f"⚠️ The name '{a_new_user}' is already taken.")
            else:
                with st.spinner("Validating and Encoding Biometric Features..."):
                    img_l = Image.open(a_reg_hand_l).convert('RGB')
                    img_r = Image.open(a_reg_hand_r).convert('RGB')
                    
                    is_l, conf_l = is_hand_image(img_l, hand_model)
                    is_r, conf_r = is_hand_image(img_r, hand_model)
                    
                    if not is_l or not is_r:
                        st.error("❌ **Validation Failed:** Detection system failed for one or both hands.")
                    else:
                        features_l = extract_encoded_features(img_l, cnn, scaler)
                        features_r = extract_encoded_features(img_r, cnn, scaler)

                        self_sim = cosine_similarity(features_l, features_r)[0][0]
                        if self_sim > 0.90:
                            st.error("⚠️ The uploaded Left and Right hand images are too similar.")
                        else:
                            duplicate_found = False
                            for existing_user, vectors in db.items():
                                if isinstance(vectors, dict) and 'left' in vectors and 'right' in vectors:
                                    sims = [cosine_similarity(features_l, vectors['left'])[0][0],
                                            cosine_similarity(features_l, vectors['right'])[0][0],
                                            cosine_similarity(features_r, vectors['left'])[0][0],
                                            cosine_similarity(features_r, vectors['right'])[0][0]]
                                    if any(s > 0.95 for s in sims):
                                        duplicate_found = True
                                        st.error("⚠️ Biometric profile already registered.")
                                        break
                            
                            if not duplicate_found:
                                db[a_new_user] = {"left": features_l, "right": features_r}
                                save_bio_db(db)
                                st.success(f"✅ Registered **{a_new_user}** successfully!")
                                st.session_state['admin_reg_seed'] += 1
                                st.rerun()

    with col2:
        st.subheader("Authentication Logs")
        df_logs = get_logs()
        if df_logs is None or df_logs.empty:
            st.info("No login activity recorded.")
        else:
            # Display as a read-only table
            st.dataframe(df_logs.iloc[::-1], use_container_width=True, hide_index=True)
            if st.button("🧹 Clear Logs"):
                if os.path.exists(LOG_FILE_PATH):
                    os.remove(LOG_FILE_PATH)
                st.rerun()

    if st.button("🚪 Logout Admin"):
        st.session_state['admin_authenticated'] = False
        st.rerun()

# --- Navigation Architecture ---

def main():
    # Top Bar Header
    col_title, col_info = st.columns([1, 1])
    with col_title:
        st.markdown("<h3 style='margin-top: 5px;'>🖐️ VeinAuth</h3>", unsafe_allow_html=True)
    with col_info:
        if st.session_state['authenticated']:
            st.markdown(f"<div class='user-info'>Logged in as: <b>{st.session_state['username']}</b></div>", unsafe_allow_html=True)

    # Navigation Tabs
    if st.session_state['authenticated']:
        tab_titles = ["Dashboard", "Home", "About", "Logout"]
        tabs = st.tabs(tab_titles)
        
        with tabs[0]:
            show_auth_system()
        with tabs[1]:
            show_home()
        with tabs[2]:
            show_about()
        with tabs[3]:
            st.markdown('<div class="stCard" style="text-align: center;">', unsafe_allow_html=True)
            st.subheader("Confirm Logout")
            st.write("Are you sure you want to end your secure biometric session?")
            if st.button("🚪 Logout Now", key="final_logout_btn"):
                st.session_state['authenticated'] = False
                st.session_state['username'] = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.session_state['admin_authenticated']:
            tab_titles = ["Dashboard", "Admin Panel", "About", "Home"]
            # Add Admin tab to the end normally or handle separately
            # For simplicity, we'll replace the Login/Register with Admin Panel
            tabs = st.tabs(["Admin Panel", "Home", "About"])
            with tabs[0]:
                show_admin_panel()
            with tabs[1]:
                show_home()
            with tabs[2]:
                show_about()
        else:
            tab_titles = ["Home", "About", "Login / Register"]
            tabs = st.tabs(tab_titles)
            
            with tabs[0]:
                show_home()
            with tabs[1]:
                show_about()
            with tabs[2]:
                show_login_register()

    # Footer
    st.markdown("<br><br><div style='text-align: center; color: #475569; font-size: 0.8rem;'>© 2026 VeinAuth Biometric System</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
