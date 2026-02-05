import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 설정 및 단위 추가 (MSE이므로 제곱 단위 표기)
labels = ['Delta Action\n($mm$)', 'Position\n($mm$)', 'Rotation\n($deg$)']
# means2 = [0.06296097, 0.12592033, 0.09215298]
# stds2 = [0.4547282, 0.90945442, 0.193879511]

# version 2
# means2 = [0.15734, 1.71198, 0.0921]
# stds2 = [0.52084, 0.42277, 0.19381]

# means2 = [0.157343151, 0.314676552, 0.558371998]
# stds2 = [0.52084832, 1.041695825, 0.94660412]

# version 2
means2 = [0.07867, 0.79955, 0.0621]
stds2 = [0.40051, 0.10308, 0.12381]

# 2. 그래프 설정
x_pos = np.arange(len(labels))

plt.figure(figsize=(15, 5))
# 3. 막대 그래프 그리기 (상단 에러 바만 설정)
# yerr=[하단 편차, 상단 편차] -> 하단은 0으로, 상단은 stds2로 설정
asymmetric_error = [means2, stds2]

bars = plt.bar(x_pos, means2, yerr=asymmetric_error, align='center', alpha=0.7, 
               color=['#4C72B0', '#55A868', '#C44E52'], 
               capsize=10, ecolor='black')

# 4. 라벨 및 타이틀 설정
plt.ylabel('Mean Squared Error (MSE)', fontsize=20)
plt.xticks(x_pos, labels, fontsize=20)

# 5. 수치 라벨링 (에러 바 끝 지점에 Mean ± Std 표시)
for i, bar in enumerate(bars):
    y_mean = means2[i]
    y_std = stds2[i]
    
    label_text = f'{y_mean:.4f} \n$\pm$ {y_std:.4f}'
    
    # 텍스트 위치: 에러 바의 끝(mean + std)보다 살짝 위
    text_y_pos = y_mean + y_std + (max(np.array(means2) + np.array(stds2)) * 0.05)
    
    plt.text(bar.get_x() + bar.get_width()/2, text_y_pos, 
             label_text, 
             ha='center', va='bottom', fontsize=18, fontweight='bold')

# Y축 범위 확보 (에러 바와 텍스트가 다 보이도록)
plt.ylim(0, max(np.array(means2) + np.array(stds2)) * 1.3)

plt.tight_layout()
plt.savefig('mse_metrics_upper_errorbar.png', dpi=300)
print("그래프가 'mse_metrics_upper_errorbar.png'로 저장되었습니다.")