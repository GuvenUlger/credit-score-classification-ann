import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import pickle
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, log_loss
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow and Keras libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1

import os
# To disable warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(False)

# Visualization settings
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

# Pandas Settings
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 1000)
pd.set_option('display.width', 1000)  # Ekran genişliği bu, koymazsan çalışmaz
pd.set_option('display.expand_frame_repr', False) # Bu da "sakın alt satıra kırma" demek

#tablo için
from tabulate import tabulate

# TEMİZ DATAYI VERDİM İNCELEME YAPTIM #
df = pd.read_csv(r"C:\Users\guven\Desktop\Credit score classification\train_v2.csv")
print(tabulate(df.head(), headers='keys', tablefmt='psql'))
df.info()
df.describe().T
print(tabulate(df.describe(include="object").T, headers='keys', tablefmt='psql'))


#ISI HARİTASI YAPIYORUM------------------------------------------------------------------------------------------------#
numeric_df = df.select_dtypes(include="number")                                                                        #
plt.figure(figsize=(10, 8))                                                                                            #
correlation_matrix = numeric_df.corr()                                                                                 #
#Plotting                                                                     Hep beraber çalıştır                     #
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')                                                #
plt.title("Correlation Heatmap with Credit Score")                                                                     #
plt.show()                                                                                                             #
#----------------------------------------------------------------------------------------------------------------------#


#Numeric columns in the DataFrame
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()


#Tablolara geçiş-------------------------------------------------------------------------------------------------------#
target_col = 'Credit_Mix'

# 2. Sayısal sütunları seç (Hedef değişkeni hariç tut)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [col for col in numeric_columns if col != target_col]

if not os.path.exists('Grafikler'):
    os.makedirs('Grafikler')

# 3. DÖNGÜ: Her bir sütun için AYRI bir tablo açıp çiziyoruz
for col in numeric_columns:
    # Her grafik için yeni ve temiz bir sayfa aç (10x6 boyutunda)
    plt.figure(figsize=(10, 6))

    # Çizim
    sns.boxplot(x=target_col, y=col, data=df, palette='viridis')

    # Süslemeler
    plt.title(f'{col} vs {target_col}', fontsize=15, fontweight='bold')
    plt.xlabel(target_col, fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    clean_col_name = col.replace('/', '_').replace(' ', '_')
    file_name = f"Grafikler/Grafik_{clean_col_name}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

#----------------------------------------------------------------------------------------------------------------------#
features = numeric_columns[:-1]

if not os.path.exists('Histogramlar'):
    os.makedirs('Histogramlar')

for feature in features:
    plt.figure(figsize=(10, 6))

    sns.histplot(data=df, x=feature, kde=True, color='blue')
    plt.title(f'Histogram of {feature}', fontsize=15)

    clean_name = feature.replace('/', '_').replace(' ', '_')
    plt.savefig(f'Histogramlar/Hist_{clean_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
#----------------------------------------------------------------------------------------------------------------------#
# Function to detect outliers
def detect_outliers_iqr(df):
    outliers = {}

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=['number'])

    for column in numeric_df.columns:
        # Calculate the first (Q1) and third quartiles (Q3)
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)

        # Calculate the IQR
        IQR = Q3 - Q1

        # Determine the lower and upper bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Identify outliers
        outlier_mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
        outliers[column] = numeric_df[column][outlier_mask]

    return outliers

# Detect outliers
outlier_results = detect_outliers_iqr(df)

# Print the results
for column, outlier_values in outlier_results.items():
    if not outlier_values.empty:
        print(f"{column} outliers:")
        print(outlier_values)
    else:
        print(f"For {column} no outliers.")
#----------------------------------------------------------------------------------------------------------------------#
df["Credit_Mix"].value_counts(normalize=True) # Standart 0.458 / Good 0.304 / Bad 0.238

#----------------------------------------------DATA PREPROCESSING------------------------------------------------------#
# List of unique loan values
unique_loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan',
                     'No Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']

# Adding a new column for each unique loan type and checking how many times it appears
for loan_type in unique_loan_types:
    # Replacing '-' and spaces with underscores, converting other characters to lowercase
    cleaned_loan_type = loan_type.replace(' ', '_').replace('-', '_').lower()

    # Counting how many times the loan_type value appears in each row
    df[cleaned_loan_type] = df['Type_of_Loan'].apply(lambda x: x.count(loan_type))
#----------------------------------------------------------------------------------------------------------------------#
df = df.drop([
    "ID", "Customer_ID", "Name", "SSN", "Type_of_Loan"], axis=1)

df.head()

# Month     Age     Occupation  Annual_Income     Monthly_Inhand_Salary  Num_Bank_Accounts  Num_Credit_Card  Interest_Rate  Num_of_Loan     Delay_from_due_date    Num_of_Delayed_Payment   Changed_Credit_Limit  Num_Credit_Inquiries  Credit_Mix   Outstanding_Debt       Credit_Utilization_Ratio     Credit_History_Age    Payment_of_Min_Amount    Total_EMI_per_month     Amount_invested_monthly                 Payment_Behaviour           Monthly_Balance     Credit_Score   Occupation_Num  Credit_Mix_Num     Payment_of_Min_Amount_Num  Payment_Behaviour_Num  student_loan
#   1     23.000    Scientist      19114.120            1824.843              3.000            4.000            3.000         4.000                3.000                   7.000                 11.270                 4.000              Good           809.980                    26.823                  265.000                    No                     49.575                   21.465                  High_spent_Small_value_payments          312.494           Good              12              -1                       1                       2                  0
#   2     23.000    Scientist      19114.120            1824.843              3.000            4.000            3.000         4.000                3.000                   4.000                 11.270                 4.000              Good           809.980                    31.945                  266.000                    No                     49.575                   21.465                   Low_spent_Large_value_payments          284.629           Good              12               1                       1                       3                  0
#   3     23.000    Scientist      19114.120            1824.843              3.000            4.000            3.000         4.000                3.000                   7.000                 11.270                 4.000              Good           809.980                    28.609                  267.000                    No                     49.575                   21.465                  Low_spent_Medium_value_payments          331.210           Good              12               1                       1                       4                  0
#   4     23.000    Scientist      19114.120            1824.843              3.000            4.000            3.000         4.000                5.000                   4.000                  6.270                 4.000              Good           809.980                    31.378                  268.000                    No                     49.575                   21.465                   Low_spent_Small_value_payments          223.451           Good              12               1                       1                       5                  0
#   5     23.000    Scientist      19114.120            1824.843              3.000            4.000            3.000         4.000                6.000                   4.000                 11.270                 4.000              Good           809.980                    24.797                  269.000                    No                     49.575                   21.465                 High_spent_Medium_value_payments          341.489           Good              12               1                       1                       1                  0
#----------------------------------------------------------------------------------------------------------------------#
payment_mapping = {
    'High_spent_Large_value_payments': 6,#Successfully managing large debts provides the most positive contribution to the credit score.
    'High_spent_Medium_value_payments': 5, #Medium-value payments with high spending positively impact the credit score.
    'High_spent_Small_value_payments': 4, #Small payments can negatively affect the credit score if debts accumulate over time.
    'Low_spent_Large_value_payments': 3, #shows quick financial responsibility, positively affecting the credit score.
    'Low_spent_Medium_value_payments': 2, #contributes positively to the credit score by demonstrating debt management.
    'Low_spent_Small_value_payments': 1 #may limit the credit history and provide minimal contribution to the credit score
}
df['Payment_Behaviour'] = df['Payment_Behaviour'].map(payment_mapping)
df['Payment_Behaviour'] = pd.to_numeric(df['Payment_Behaviour'], downcast='integer')

# Convert the credit_mix column to numerical values
df['Credit_Mix'] = df['Credit_Mix'].map({'Good': 2, 'Standard': 1, 'Bad': 0})
df['Credit_Mix'] = pd.to_numeric(df['Credit_Mix'], downcast='integer')

# Convert the payment_of_min_amount column to numerical values
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})
df['Payment_of_Min_Amount'] = pd.to_numeric(df['Payment_of_Min_Amount'], downcast='integer')

df = pd.get_dummies(df, columns=['Occupation'])

month_map = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8
}
#Mapping
df['Month'] = df['Month'].map(month_map)
df['Month'] = pd.to_numeric(df['Month'], downcast='integer')
# Separate features and target variable
X = df.drop("Credit_Mix", axis=1)
y = df.Credit_Mix
#-------------------------------------#
y.value_counts(normalize=True) # unbalanced data  2:Good, 1: Standard, 0: Poor
df.head()

#---------------------------------------------TRAIN-TEST SPLITS--------------------------------------------------------#
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder

robust_columns = [
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
    'Annual_Income', 'Monthly_Inhand_Salary',
    'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit'
]

drop_listesi = [
    'Name', 'Occupation', 'Type_of_Loan', 'SSN', 'ID', 'Customer_ID', 'Month', 'Age',
    'Credit_Mix', 'Credit_Mix_Good', 'Payment_Behaviour', 'Payment_of_Min_Amount',
    'Credit_Score'
]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# 3. Scaling Setup
X_train_numeric = X_train.select_dtypes(include=[np.number])
standard_columns = [col for col in X_train_numeric.columns
                    if col not in robust_columns
                    and col not in drop_listesi]

scaler = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), standard_columns),
        ('robust', RobustScaler(), robust_columns)
    ],
    remainder='drop'
)

# 4. Apply Scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Target Encoding (Important for ANN)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 6. Class Weights
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded),
                                           y=y_train_encoded)
class_weights = {i: weight for i, weight in zip(np.unique(y_train_encoded), class_weights_array)}

# 7. Build Model
tf.random.set_seed(42)

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Correct input shape

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(512, activation='relu', kernel_regularizer=l1(1e-4)),
    BatchNormalization(),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(3, activation='softmax')
])

# 8. Compile
model.compile(optimizer=Adam(learning_rate=0.0003),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 9. Train
early_stopping = EarlyStopping(monitor='val_accuracy', patience=35, restore_best_weights=True)

history = model.fit(x=X_train_scaled,
                    y=y_train_encoded,
                    validation_data=(X_test_scaled, y_test_encoded),
                    batch_size=1024,
                    epochs=500,
                    verbose=1,
                    callbacks=[early_stopping],
                    class_weight=class_weights)
#----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------FINAL MODEL PREDICTION-------------------------------------------------------#
pickle.dump(scaler, open("credit_score_scaler.pkl", "wb"))

if 'le' in locals():
    pickle.dump(le, open("credit_score_le.pkl", "wb"))
    print("-> Label Encoder kaydedildi: credit_score_le.pkl")
else:
    print("UYARI: 'le' (LabelEncoder) bulunamadı! ANN eğitim kısmını çalıştırmadıysan çalıştır!")

model.save('final_model_credit_score.keras')
model.summary()
#--load model--#
final_model = load_model('final_model_credit_score.keras')
#--------------------------------------------------------------------#
# Generating random data suitable for statistics using your sample dataset
def generate_synthetic_data(df, num_samples=50):
    synthetic_data = pd.DataFrame()

    numeric_df = df.select_dtypes(include=[np.number])

    for column in numeric_df.columns:
        if column != 'credit_score':

            min_val = numeric_df[column].min()
            max_val = numeric_df[column].max()
            mean_val = numeric_df[column].mean()
            std_val = numeric_df[column].std()

            if pd.isna(std_val): std_val = 0
            if pd.isna(mean_val): mean_val = 0

            synthetic_data[column] = np.random.normal(loc=mean_val, scale=std_val, size=num_samples)
            synthetic_data[column] = synthetic_data[column].clip(lower=min_val, upper=max_val)

    return synthetic_data

# Generating 50 rows of synthetic data based on the statistics of your current dataset
df_prediction = generate_synthetic_data(df, num_samples=50)
#------#
df_prediction_scaled = scaler.transform(df_prediction)
final_model.predict(df_prediction_scaled)
#----------------------------------------------------------------------------------------------------------------------#
y_pred_probabilities = final_model.predict(df_prediction_scaled)
normalized_predictions = tf.nn.softmax(y_pred_probabilities, axis=-1).numpy()#normalized probabilites of each class
#------#
y_pred = np.argmax(y_pred_probabilities, axis=1)
pred_df = pd.DataFrame({'pred': y_pred})

# Mapping dictionary for the reverse transformation
mapping = {2: 'Good', 1: 'Standard', 0: 'Poor'}

# Apply the mapping
pred_df["pred"] = pred_df["pred"] .map(mapping)
#---------------------------------------------------------------#
pred_df["pred_proba_poor"] = normalized_predictions[:,0]
pred_df["pred_proba_standard"] = normalized_predictions[:,1]
pred_df["pred_proba_good"] = normalized_predictions[:,2]
pred_df

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.utils import to_categorical

class_names = ['Good', 'Poor', 'Standard']

try:
    y_test_encoded = y_test.astype(int)
except:
    y_test_encoded = y_test

y_test_ann = to_categorical(y_test_encoded, num_classes=3)

y_pred_probs_test = final_model.predict(X_test_scaled, verbose=0)
y_pred_test_classes = np.argmax(y_pred_probs_test, axis=1)

# --- GRAFİKLER ---
plt.figure(figsize=(14, 6))

# GRAFİK 1: CONFUSION MATRIX
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test_encoded, y_pred_test_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Kim Kimi Karıştırdı?)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Olan')

# GRAFİK 2: ROC CURVE (Sınıf Ayırt Etme Gücü)
plt.subplot(1, 2, 2)

# Y test'i binarize et (ROC için lazım)
y_test_bin = label_binarize(y_test_encoded, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

colors = ['green', 'red', 'blue'] # Good(Yeşil), Poor(Kırmızı), Standard(Mavi)
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs_test[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label='ROC: {0} (AUC = {1:0.2f})'.format(class_names[i], roc_auc))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm)')
plt.ylabel('True Positive Rate (Doğru Tespit)')
plt.title('ROC Eğrileri')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model

print("--- EKSİK PARÇALAR TAMAMLANIYOR & GRAFİK ÇİZİLİYOR ---")

# 1. ÖNCE GEREKLİLERİ YÜKLE (Hafızada yoksa diye garanti olsun)
if 'final_model' not in locals():
    final_model = load_model('final_model_credit_score.keras')
if 'scaler_yuklenen' not in locals():
    scaler_yuklenen = pickle.load(open("credit_score_scaler.pkl", "rb"))
if 'le_yuklenen' not in locals():
    le_yuklenen = pickle.load(open("credit_score_le.pkl", "rb"))

# 2. EKSİK OLAN O "PREDICTIONS" DEĞİŞKENLERİNİ HESAPLA
# (df_prediction_scaled'in hafızada olduğunu varsayıyorum. Yoksa önce veri üretme kodunu çalıştır!)
try:
    # A. Olasılıkları Hesapla (predictions_prob)
    print("Tahminler yapılıyor...")
    predictions_prob = final_model.predict(df_prediction_scaled, verbose=0)

    # B. Sınıfı Seç (0, 1, 2)
    predictions_classes = np.argmax(predictions_prob, axis=1)

    # C. İnsan Diline Çevir (decoded_predictions -> Good, Standard...)
    decoded_predictions = le_yuklenen.inverse_transform(predictions_classes)

    print("Değişkenler oluşturuldu. Şimdi grafik geliyor...")

except NameError:
    print("HATA: 'df_prediction_scaled' bulunamadı! Önce sentetik veri üreten kodu çalıştır!")
    # Hata almamak için çıkış yapıyoruz
    raise

# ---------------------------------------------------------
# 3. SENİN İSTEDİĞİN GRAFİK KODU (ARTIK ÇALIŞIR)
# ---------------------------------------------------------

print("--- 2. BÖLÜM: SENTETİK TAHMİN ANALİZİ ---")

# Veriyi Hazırla
results_df = pd.DataFrame({
    'Tahmin': decoded_predictions,
    'Güven': np.max(predictions_prob, axis=1)  # En yüksek olasılık
})

plt.figure(figsize=(14, 6))

# GRAFİK 3: PASTA GRAFİĞİ
plt.subplot(1, 2, 1)
counts = results_df['Tahmin'].value_counts()
# Renkleri sınıf sayısına göre ayarlayalım
colors = ['#ff9999', '#66b3ff', '#99ff99'] if len(counts) == 3 else None

plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Modelin Tahmin Dağılımı (Kime Ne Dedi?)')

# GRAFİK 4: GÜVEN HİSTOGRAMI
plt.subplot(1, 2, 2)
sns.histplot(results_df['Güven'], bins=20, kde=True, color='purple')
# Ortalama çizgisini ekle
plt.axvline(results_df['Güven'].mean(), color='red', linestyle='dashed', linewidth=1)
plt.title(f"Modelin Kendine Güveni (Ortalama: %{results_df['Güven'].mean() * 100:.1f})")
plt.xlabel('Olasılık Değeri (0-1 arası)')

plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------------------------------------------------#
analysis_df = df_prediction.copy()
analysis_df['Tahmin_Sinifi'] = decoded_predictions

features_to_check = ['Outstanding_Debt', 'Interest_Rate', 'Annual_Income', 'Total_EMI_per_month']

plt.figure(figsize=(15, 10))

for i, col in enumerate(features_to_check):
    if col in analysis_df.columns:
        plt.subplot(2, 2, i+1)
        # Boxplot: Hangi sınıfın ortalaması nerede?
        sns.boxplot(x='Tahmin_Sinifi', y=col, data=analysis_df, palette="Set2")
        plt.title(f"{col} vs Tahmin Edilen Sınıf")
    else:
        print(f"UYARI: {col} sütunu sentetik veride bulunamadı, çizilemedi.")

plt.tight_layout()
plt.show()