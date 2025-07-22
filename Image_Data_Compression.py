import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# 또는: 'NanumGothic', 'AppleGothic' (macOS), 'Noto Sans CJK KR' (Linux)

class JPEGCompressionTool:
    def __init__(self):
        self.original_image = None
        self.compressed_image = None
        
    def compress_with_pil(self, image_path, quality=85, optimize=True):
        """PIL을 사용한 JPEG 압축"""
        try:
            # 원본 이미지 로드
            with Image.open(image_path) as img:
                # RGB 모드로 변환 (JPEG는 RGB만 지원)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 메모리 버퍼에 JPEG로 저장
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=optimize)
                
                # 압축된 이미지 다시 로드
                buffer.seek(0)
                compressed_img = Image.open(buffer)
                
                return compressed_img, buffer.getvalue()
        
        except Exception as e:
            print(f"압축 오류: {e}")
            return None, None
    
    def get_compression_info(self, original_path, compressed_data):
        """압축 정보 계산"""
        original_size = os.path.getsize(original_path)
        compressed_size = len(compressed_data)
        
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'size_reduction': original_size - compressed_size
        }
    
    def calculate_image_quality_metrics(self, original_img, compressed_img):
        """이미지 품질 지표 계산"""
        # 배열로 변환
        original_array = np.array(original_img)
        compressed_array = np.array(compressed_img)
        
        # 원본 이미지가 RGBA면 RGB로 변환
        if original_array.shape[-1] == 4:
            # 알파 채널 제거하고 RGB만 사용
            original_array = original_array[:, :, :3]
        
        # 압축된 이미지도 RGB로 확인
        if compressed_array.shape[-1] == 4:
            compressed_array = compressed_array[:, :, :3]
        
        # 크기가 다르면 맞춤
        if original_array.shape != compressed_array.shape:
            print(f"크기 불일치: 원본{original_array.shape}, 압축{compressed_array.shape}")
            # 더 작은 크기로 맞춤
            min_h = min(original_array.shape[0], compressed_array.shape[0])
            min_w = min(original_array.shape[1], compressed_array.shape[1])
            original_array = original_array[:min_h, :min_w, :3]
            compressed_array = compressed_array[:min_h, :min_w, :3]
        
        # MSE (Mean Squared Error) 계산
        mse = np.mean((original_array.astype(np.float32) - compressed_array.astype(np.float32)) ** 2)
        
        # PSNR (Peak Signal-to-Noise Ratio) 계산
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        return {
            'mse': mse,
            'psnr': psnr
        }
    
    def batch_compress_images(self, input_folder, output_folder, quality=85):
        """폴더 내 모든 이미지 일괄 압축"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        results = []
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, 
                                         os.path.splitext(filename)[0] + '_compressed.jpg')
                
                try:
                    # 이미지 압축
                    compressed_img, compressed_data = self.compress_with_pil(
                        input_path, quality=quality
                    )
                    
                    if compressed_img:
                        # 압축된 이미지 저장
                        compressed_img.save(output_path, 'JPEG')
                        
                        # 압축 정보 계산
                        info = self.get_compression_info(input_path, compressed_data)
                        info['filename'] = filename
                        info['output_path'] = output_path
                        
                        results.append(info)
                        print(f"✓ {filename} 압축 완료 - {info['compression_ratio']:.1f}% 압축")
                    
                except Exception as e:
                    print(f"✗ {filename} 압축 실패: {e}")
        
        return results
    
    def create_comparison_plot(self, original_path, quality_levels=[10, 30, 50, 70, 90]):
        """다양한 품질 레벨에서의 압축 결과 비교"""
        fig, axes = plt.subplots(2, len(quality_levels), figsize=(15, 6))
        
        # 원본 이미지를 RGB로 변환
        original_img = Image.open(original_path)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        original_size = os.path.getsize(original_path)
        
        for i, quality in enumerate(quality_levels):
            # 압축 수행
            compressed_img, compressed_data = self.compress_with_pil(
                original_path, quality=quality
            )
            
            if compressed_img:
                # 압축 정보 계산
                info = self.get_compression_info(original_path, compressed_data)
                quality_metrics = self.calculate_image_quality_metrics(
                    original_img, compressed_img
                )
                
                # 이미지 표시
                axes[0, i].imshow(compressed_img)
                axes[0, i].set_title(f'Quality {quality}')
                axes[0, i].axis('off')
                
                # 정보 표시
                info_text = f"Size: {info['compressed_size']//1024}KB\n"
                info_text += f"Ratio: {info['compression_ratio']:.1f}%\n"
                info_text += f"PSNR: {quality_metrics['psnr']:.1f}dB"
                
                axes[1, i].text(0.1, 0.5, info_text, fontsize=20, 
                               verticalalignment='center')
                axes[1, i].axis('off')
        
        plt.suptitle(f'JPEG 압축 품질 비교 (원본: {original_size//1024}KB)')
        plt.tight_layout()
        plt.show()
    
    def analyze_compression_artifacts(self, original_path, compressed_path):
        """압축 아티팩트 분석"""
        original_img = np.array(Image.open(original_path))
        compressed_img = np.array(Image.open(compressed_path))
        
        # 차이 이미지 계산
        diff = np.abs(original_img.astype(np.float32) - compressed_img.astype(np.float32))
        
        # 블록 아티팩트 검출 (8x8 블록 경계에서의 불연속성)
        def detect_block_artifacts(img):
            artifacts = np.zeros_like(img[:, :, 0])
            
            # 수직 경계 검사
            for i in range(7, img.shape[0], 8):
                if i < img.shape[0] - 1:
                    vertical_diff = np.abs(img[i, :, 0].astype(np.float32) - 
                                         img[i+1, :, 0].astype(np.float32))
                    artifacts[i, :] = vertical_diff
            
            # 수평 경계 검사
            for j in range(7, img.shape[1], 8):
                if j < img.shape[1] - 1:
                    horizontal_diff = np.abs(img[:, j, 0].astype(np.float32) - 
                                           img[:, j+1, 0].astype(np.float32))
                    artifacts[:, j] = horizontal_diff
            
            return artifacts
        
        block_artifacts = detect_block_artifacts(compressed_img)
        
        # 결과 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(compressed_img)
        axes[0, 1].set_title('압축된 이미지')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(diff.astype(np.uint8))
        axes[1, 0].set_title('차이 이미지')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(block_artifacts, cmap='hot')
        axes[1, 1].set_title('블록 아티팩트')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'mean_difference': np.mean(diff),
            'max_difference': np.max(diff),
            'block_artifact_score': np.mean(block_artifacts)
        }

class JPEGCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG 압축 도구")
        self.root.geometry("600x400")
        
        self.compression_tool = JPEGCompressionTool()
        self.current_image_path = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 파일 선택
        ttk.Button(main_frame, text="이미지 선택", 
                  command=self.select_image).grid(row=0, column=0, pady=5)
        
        self.file_label = ttk.Label(main_frame, text="선택된 파일: 없음")
        self.file_label.grid(row=0, column=1, padx=10, pady=5)
        
        # 품질 설정
        ttk.Label(main_frame, text="압축 품질:").grid(row=1, column=0, pady=5)
        self.quality_var = tk.IntVar(value=50)
        quality_scale = ttk.Scale(main_frame, from_=1, to=100, 
                                 variable=self.quality_var, orient=tk.HORIZONTAL)
        quality_scale.grid(row=1, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))
        
        self.quality_label = ttk.Label(main_frame, text="85")
        self.quality_label.grid(row=1, column=2, pady=5)
        
        # 품질 값 업데이트
        quality_scale.configure(command=self.update_quality_label)
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="압축 수행", 
                  command=self.compress_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="품질 비교", 
                  command=self.compare_qualities).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="일괄 압축", 
                  command=self.batch_compress).pack(side=tk.LEFT, padx=5)
        
        # 결과 표시
        self.result_text = tk.Text(main_frame, height=15, width=70)
        self.result_text.grid(row=3, column=0, columnspan=3, pady=10)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, 
                                 command=self.result_text.yview)
        scrollbar.grid(row=3, column=3, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
    
    def update_quality_label(self, value):
        self.quality_label.configure(text=str(int(float(value))))
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.file_label.configure(text=f"선택된 파일: {os.path.basename(file_path)}")
    
    def compress_image(self):
        if not self.current_image_path:
            messagebox.showerror("오류", "먼저 이미지를 선택하세요.")
            return
        
        try:
            quality = self.quality_var.get()
            
            # 압축 수행
            compressed_img, compressed_data = self.compression_tool.compress_with_pil(
                self.current_image_path, quality=quality
            )
            
            if compressed_img:
                # 결과 저장
                output_path = os.path.splitext(self.current_image_path)[0] + f"_compressed_q{quality}.jpg"
                compressed_img.save(output_path)
                
                # 정보 계산
                info = self.compression_tool.get_compression_info(
                    self.current_image_path, compressed_data
                )
                
                # 원본 이미지와 품질 비교
                original_img = Image.open(self.current_image_path)
                quality_metrics = self.compression_tool.calculate_image_quality_metrics(
                    original_img, compressed_img
                )
                
                # 결과 표시
                result = f"=== 압축 완료 ===\n"
                result += f"출력 파일: {output_path}\n"
                result += f"원본 크기: {info['original_size']//1024} KB\n"
                result += f"압축 크기: {info['compressed_size']//1024} KB\n"
                result += f"압축률: {info['compression_ratio']:.1f}%\n"
                result += f"용량 절약: {info['size_reduction']//1024} KB\n"
                result += f"PSNR: {quality_metrics['psnr']:.2f} dB\n"
                result += f"MSE: {quality_metrics['mse']:.2f}\n\n"
                
                self.result_text.insert(tk.END, result)
                self.result_text.see(tk.END)
                
                messagebox.showinfo("완료", f"압축이 완료되었습니다!\n저장 위치: {output_path}")
        
        except Exception as e:
            messagebox.showerror("오류", f"압축 중 오류가 발생했습니다: {e}")
    
    def compare_qualities(self):
        if not self.current_image_path:
            messagebox.showerror("오류", "먼저 이미지를 선택하세요.")
            return
        
        try:
            self.compression_tool.create_comparison_plot(self.current_image_path)
        except Exception as e:
            messagebox.showerror("오류", f"비교 중 오류가 발생했습니다: {e}")
    
    def batch_compress(self):
        input_folder = filedialog.askdirectory(title="입력 폴더 선택")
        if not input_folder:
            return
        
        output_folder = filedialog.askdirectory(title="출력 폴더 선택")
        if not output_folder:
            return
        
        try:
            quality = self.quality_var.get()
            results = self.compression_tool.batch_compress_images(
                input_folder, output_folder, quality=quality
            )
            
            # 결과 표시
            result_text = f"=== 일괄 압축 완료 ===\n"
            result_text += f"처리된 파일 수: {len(results)}\n\n"
            
            total_original = sum(r['original_size'] for r in results)
            total_compressed = sum(r['compressed_size'] for r in results)
            total_ratio = (1 - total_compressed / total_original) * 100
            
            result_text += f"전체 원본 크기: {total_original//1024} KB\n"
            result_text += f"전체 압축 크기: {total_compressed//1024} KB\n"
            result_text += f"전체 압축률: {total_ratio:.1f}%\n\n"
            
            for result in results:
                result_text += f"{result['filename']}: {result['compression_ratio']:.1f}% 압축\n"
            
            self.result_text.insert(tk.END, result_text)
            self.result_text.see(tk.END)
            
            messagebox.showinfo("완료", f"일괄 압축이 완료되었습니다!\n{len(results)}개 파일 처리됨")
        
        except Exception as e:
            messagebox.showerror("오류", f"일괄 압축 중 오류가 발생했습니다: {e}")

# 실행
if __name__ == "__main__":
    # 콘솔에서 사용하는 경우
    print("JPEG 압축 도구")
    print("1. GUI 실행")
    print("2. 콘솔에서 압축")
    
    choice = input("선택하세요 (1 또는 2): ")
    
    if choice == "1":
        # GUI 실행
        root = tk.Tk()
        app = JPEGCompressionGUI(root)
        root.mainloop()
    
    elif choice == "2":
        # 콘솔에서 압축
        tool = JPEGCompressionTool()
        
        image_path = input("이미지 경로를 입력하세요: ")
        if not os.path.exists(image_path):
            print("파일을 찾을 수 없습니다.")
        else:
            quality = int(input("압축 품질을 입력하세요 (1-100): "))
            
            compressed_img, compressed_data = tool.compress_with_pil(
                image_path, quality=quality
            )
            
            if compressed_img:
                output_path = os.path.splitext(image_path)[0] + f"_compressed_q{quality}.jpg"
                compressed_img.save(output_path)
                
                info = tool.get_compression_info(image_path, compressed_data)
                
                print(f"\n압축 완료!")
                print(f"출력 파일: {output_path}")
                print(f"원본 크기: {info['original_size']//1024} KB")
                print(f"압축 크기: {info['compressed_size']//1024} KB")
                print(f"압축률: {info['compression_ratio']:.1f}%")
                print(f"용량 절약: {info['size_reduction']//1024} KB")
                
                # 품질 비교 그래프 표시
                show_comparison = input("품질 비교 그래프를 보시겠습니까? (y/n): ")
                if show_comparison.lower() == 'y':
                    tool.create_comparison_plot(image_path)
            else:
                print("압축에 실패했습니다.")
    
    else:
        print("잘못된 선택입니다.")