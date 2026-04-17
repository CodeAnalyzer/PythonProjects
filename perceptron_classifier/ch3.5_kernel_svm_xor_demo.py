import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Visualization of decision regions for any classifier."""
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Построение поверхности решений
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Отображение образцов классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Класс {cl}', 
                    edgecolor='black')

print("XOR: KLASICHESKAYA PROBLEMA NELINEYNOY KLASIFIKACII")
print("=" * 60)

# 1. Sozdanie sinteticheskogo nabora dannikh XOR
print("\n1. SOZDANIE NABORA DANNIKH XOR")
print("-" * 40)
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)

print(f'Forma X_xor: {X_xor.shape}')
print(f'Forma y_xor: {y_xor.shape}')
print(f'Klassy: {np.unique(y_xor)}')
print(f'Raspredelenie klassov: {np.bincount(y_xor)}')

# 2. Vizualizaciya iskhodnykh dannykh
print("\n2. VIZUALIZACIYA ISKHODNYKH DANNIKH")
print("-" * 40)

plt.figure(figsize=(12, 4))

# Iskhodnye dannye
plt.subplot(1, 3, 1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='royalblue', marker='s', label='Class 1', alpha=0.8)
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
            c='tomato', marker='o', label='Class 0', alpha=0.8)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Priznak 1')
plt.ylabel('Priznak 2')
plt.title('XOR: Iskhodnye dannye')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# 3. Lineynaya SVM (ne spravitsya)
print("\n3. LINEYNAYA SVM - PROVAL")
print("-" * 40)

svm_linear = SVC(kernel='linear', random_state=1, C=1.0)
svm_linear.fit(X_xor, y_xor)

accuracy_linear = svm_linear.score(X_xor, y_xor)
print(f'Tochnost lineynoy SVM: {accuracy_linear:.3f} ({accuracy_linear*100:.1f}%)')
print(f'Kolichestvo opornykh vektorov: {svm_linear.n_support_.sum()}')

plt.subplot(1, 3, 2)
plot_decision_regions(X_xor, y_xor, svm_linear)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Priznak 1')
plt.ylabel('Priznak 2')
plt.title(f'Lineynaya SVM\nTochnost: {accuracy_linear*100:.1f}%')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# 4. Yadernaya SVM s RBF (radial-basis-function)
print("\n4. YADERNAYA SVM S RBF - USPEKH")
print("-" * 40)

svm_rbf = SVC(kernel='rbf', random_state=1, C=1.0, gamma=0.3)
svm_rbf.fit(X_xor, y_xor)

accuracy_rbf = svm_rbf.score(X_xor, y_xor)
print(f'Tochnost RBF SVM: {accuracy_rbf:.3f} ({accuracy_rbf*100:.1f}%)')
print(f'Kolichestvo opornykh vektorov: {svm_rbf.n_support_.sum()}')
print(f'Parametr gamma: {svm_rbf.gamma}')
print(f'Parametr C: {svm_rbf.C}')

plt.subplot(1, 3, 3)
plot_decision_regions(X_xor, y_xor, svm_rbf)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Priznak 1')
plt.ylabel('Priznak 2')
plt.title(f'RBF SVM (gamma=0.3)\nTochnost: {accuracy_rbf*100:.1f}%')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Eksperiment s raznymi parametrami gamma
print("\n5. EKSPERIMENT S PARAMETROM GAMMA")
print("-" * 40)

gamma_values = [0.1, 0.3, 1.0, 10.0]

plt.figure(figsize=(15, 4))
for i, gamma in enumerate(gamma_values):
    svm = SVC(kernel='rbf', random_state=1, C=1.0, gamma=gamma)
    svm.fit(X_xor, y_xor)
    accuracy = svm.score(X_xor, y_xor)
    
    plt.subplot(1, 4, i+1)
    plot_decision_regions(X_xor, y_xor, svm)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('Priznak 1')
    plt.ylabel('Priznak 2')
    plt.title(f'Gamma = {gamma}\nTochnost: {accuracy*100:.1f}%\nOpornye: {svm.n_support_.sum()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Eksperiment s raznymi parametrami C
print("\n6. EKSPERIMENT S PARAMETROM C")
print("-" * 40)

c_values = [0.1, 1.0, 10.0, 100.0]

plt.figure(figsize=(15, 4))
for i, C in enumerate(c_values):
    svm = SVC(kernel='rbf', random_state=1, C=C, gamma=0.3)
    svm.fit(X_xor, y_xor)
    accuracy = svm.score(X_xor, y_xor)
    
    plt.subplot(1, 4, i+1)
    plot_decision_regions(X_xor, y_xor, svm)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('Priznak 1')
    plt.ylabel('Priznak 2')
    plt.title(f'C = {C}\nTochnost: {accuracy*100:.1f}%\nOpornye: {svm.n_support_.sum()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Teoreticheskoe ob'yasnenie
print("\n7. TEORETICHESKOE OBYASNENIE")
print("-" * 40)

print("PROBLEMA XOR:")
print(" XOR - klassicheskiy primer lineyno nerazdelimykh dannykh")
print(" Klas 1: (x1 > 0 XOR x2 > 0) - protivopolozhnye kvadranty")
print(" Klas 0: (x1 <= 0 XOR x2 <= 0) - drugie protivopolozhnye kvadranty")
print(" Lineynaya granitsa ne mozhet razdelit' eti klassy")

print("\nYADERNYY TRYUK (KERNEL TRICK):")
print(" Ideya: proektirovat' dannye v prostranstvo vysshey razmernosti")
print(" V novom prostranstve dannye mogut stat' lineyno razdelimymi")
print(" RBF yadro: K(x, x') = exp(-gamma * ||x - x'||²)")

print("\nPARAMETR GAMMA:")
print(" Opredelyaet 'radius vliyaniya' odnoy obuchayushchey tochki")
print(" Maloe gamma: shirokiy radius, gladkaya granitsa")
print(" Bol'shoe gamma: uzkiy radius, izvilistaya granitsa, risk pereobucheniya")

print("\nPARAMETR C:")
print(" Regulyarizaciya, kak v lineynoy SVM")
print(" Maloe C: shirokiy otstup, dopustimyi oshibki")
print(" Bol'shoe C: uzkiy otstup, menshe oshibok obucheniya")

print("\nRBF YADRO:")
print(" Samoe populyarnoe yadro dlya SVM")
print(" Universal'noye priblizhenie (universal approximation theorem)")
print(" Rabotaet dlya lyubykh dannykh, no trebuyet podbora parametrov")

print("\nPRAKTIChESKIE REKOMENDACII:")
print(" 1. Nachnite s gamma = 'scale' (avtomaticheskiy podbor)")
print(" 2. Ispol'zuyte GridSearchCV dlya podbora C i gamma")
print(" 3. Standartizuyte dannye pered RBF SVM")
print(" 4. Sledite za pereobucheniem pri bol'shikh gamma i C")

print("\nSRavnenie YADER:")
print(" Linear: dlya lineyno razdelimykh dannykh")
print(" Polynomial: dlya slozhnyh nelineynyh zavisimostey")
print(" RBF (Gaussian): universal'noye, luchshe dlya bol'shinstva zadach")
print(" Sigmoid: redko ispol'zuetsya, podobro neural'noy seti")

# 8. Analiz rezul'tatov
print("\n8. ANALIZ REZUL'TATOV")
print("-" * 40)

print(f"Lineynaya SVM: {accuracy_linear*100:.1f}% tochnost")
print(f"RBF SVM (gamma=0.3): {accuracy_rbf*100:.1f}% tochnost")
print(f"Uluchsheniye: {(accuracy_rbf - accuracy_linear)*100:.1f}% punktov")

print(f"\nKlyuchevoy vyvod:")
print(f" Yadernaya SVM reshayet problemu, nereshimuyu dlya lineynykh metodov")
print(f" XOR - klassicheskiy primer, gde yadernyy triuk ochen effektiven")

print("\n" + "="*60)
print("YADERNAYA SVM - MOSHCHNYY INSTRUMENT Dlya NELINEYNOY KLASIFIKACII")
print("="*60)
