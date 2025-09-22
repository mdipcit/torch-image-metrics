# torch-image-metrics - Claude Code プロジェクトガイド

## プロジェクト概要

**torch-image-metrics**は、PyTorchベースの統一画像品質評価メトリクスライブラリです。
現在、[Generative-Latent-Optimization](~/ghq/github.com/mdipcit/Generative-Latent-Optimization)プロジェクトのメトリクス処理部分を独立したPyPIパッケージとして分離する初期段階にあります。

### 主な目的
- 画像品質評価メトリクスの統一API提供
- PSNR、SSIM、LPIPS、FIDなどの主要メトリクス実装
- バッチ処理とGPU最適化のサポート
- 簡単に使えるPythonパッケージとしての配布

## 開発環境セットアップ

### 必要な環境
- Python 3.10 (元のプロジェクトと同じバージョン)
- uv (パッケージ管理ツール)
- CUDA対応GPU (オプション、メトリクス計算の高速化用)

### 初期セットアップ手順
```bash
# 1. Python 3.10の設定
echo "3.10" > .python-version

# 2. 仮想環境の作成と有効化
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows

# 3. 依存関係のインストール
uv sync

# 4. 開発ツールのセットアップ
uv add --dev pytest pytest-mock ruff mypy

# 5. pre-commitのインストール (オプション)
uv add --dev pre-commit
pre-commit install
```

## 主要コマンド

### テスト実行
```bash
# 全テストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_basic.py

# カバレッジレポート付きでテスト実行
pytest --cov=src/torch_image_metrics --cov-report=term-missing

# 失敗時に即座に停止
pytest -x

# 詳細出力でテスト実行
pytest -v

# 特定のテスト関数のみ実行
pytest -k test_psnr
```

### コード品質チェック
```bash
# コードの品質チェック（自動修正付き）
ruff check src/ --fix

# コードフォーマット
ruff format src/

# 型チェック
mypy src/torch_image_metrics/

# 全品質チェックを一度に実行
ruff check src/ --fix && ruff format src/ && mypy src/torch_image_metrics/
```

### ビルドとパッケージング
```bash
# パッケージのビルド（wheel と sdist を作成）
uv build

# ローカルでのインストールテスト
uv pip install -e .

# PyPIへの公開（認証情報の設定が必要）
uv publish

# テスト用PyPIへの公開
uv publish --repository testpypi
```

### 依存関係管理
```bash
# 依存関係の更新
uv sync

# 新しいパッケージの追加
uv add numpy torch torchvision

# 開発用パッケージの追加
uv add --dev pytest ruff

# 依存関係の確認
uv pip list
```

## プロジェクト構造

### 目標とするディレクトリ構造
```
torch-image-metrics/
├── .python-version          # Python バージョン指定 (3.10)
├── pyproject.toml          # プロジェクト設定とメタデータ
├── README.md               # プロジェクトドキュメント
├── CLAUDE.md               # Claude Code用ガイド（このファイル）
├── LICENSE                 # MITライセンス
├── src/
│   └── torch_image_metrics/
│       ├── __init__.py     # パッケージエントリーポイント
│       ├── evaluator.py    # 統一評価器 (tim.Evaluator)
│       ├── calculator.py   # 統一計算器 (tim.Calculator)
│       ├── metrics/        # 個別メトリクス実装
│       │   ├── __init__.py
│       │   ├── basic.py    # PSNR, SSIM
│       │   ├── perceptual.py # LPIPS
│       │   └── dataset.py  # FID
│       ├── core/           # 基盤クラス
│       │   ├── __init__.py
│       │   ├── base_metric.py
│       │   └── data_structures.py
│       └── utils/          # ユーティリティ関数
│           ├── __init__.py
│           ├── image_utils.py
│           └── tensor_utils.py
├── tests/
│   ├── unit/              # ユニットテスト
│   │   ├── test_basic_metrics.py
│   │   ├── test_perceptual_metrics.py
│   │   └── test_dataset_metrics.py
│   └── integration/       # 統合テスト
│       └── test_evaluator.py
└── examples/              # 使用例
    ├── basic_usage.py
    └── batch_evaluation.py
```

## 移行計画

### Phase 1: コアコンポーネントの移行（現在のフェーズ）

#### 移行元ファイル（Generative-Latent-Optimization）
```
src/generative_latent_optimization/
├── metrics/
│   ├── image_metrics.py        → basic.py (PSNR, SSIM)
│   ├── individual_metrics.py   → perceptual.py (LPIPS)
│   ├── dataset_metrics.py      → dataset.py (FID)
│   └── unified_calculator.py   → calculator.py
└── evaluation/
    └── simple_evaluator.py      → evaluator.py
```

#### 移行手順
```bash
# 1. メトリクスコードの特定
cd ~/ghq/github.com/mdipcit/Generative-Latent-Optimization
grep -r "class.*Metric" src/generative_latent_optimization/metrics/

# 2. 依存関係の分析
grep -r "import" src/generative_latent_optimization/metrics/ | grep -v "generative_latent_optimization"

# 3. コードのコピーと適応
# 各メトリクスファイルを新しい構造に合わせて移行

# 4. テストの作成
# 各メトリクスに対応するテストを作成
```

### Phase 2: API設計と実装

#### 主要APIクラス
```python
# tim.Evaluator - データセット評価用
evaluator = tim.Evaluator(metrics=['psnr', 'ssim', 'lpips'])
results = evaluator.evaluate_dataset(test_dir, ref_dir)

# tim.Calculator - 個別計算用
calc = tim.Calculator()
metrics = calc.compute_all_metrics(img1, img2)

# クイックAPI - 単一メトリクス計算
psnr_value = tim.quick_psnr(img1, img2)
ssim_value = tim.quick_ssim(img1, img2)
```

### Phase 3: PyPIパッケージ公開

#### pyproject.toml設定
```toml
[project]
name = "torch-image-metrics"
version = "0.1.0"
description = "Unified PyTorch image quality metrics library"
authors = [
    { name = "Yus314", email = "shizhaoyoujie@gmail.com" }
]
requires-python = ">=3.10, <3.12"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pillow>=9.0.0",
    "numpy>=1.21.0",
    "scipy>=1.10.0",
]

[project.optional-dependencies]
full = ["lpips>=0.1.4", "pytorch-fid>=0.3.0"]
dev = ["pytest>=8.0.0", "pytest-mock>=3.14.0", "ruff>=0.1.0", "mypy>=1.0.0"]
```

## 実装ガイドライン（元プロジェクト分析に基づく）

### 基底クラスの使用

元のプロジェクトでは`BaseMetric`クラスが定義されています。新しいメトリクスを実装する際は、この設計パターンを踏襲します：

```python
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import torch

class BaseMetric(ABC):
    """全メトリクスの基底クラス"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: 計算に使用するデバイス ('cuda' or 'cpu')
        """
        self.device = self._validate_and_setup_device(device)
        self._setup_metric_specific_resources()
    
    def _validate_and_setup_device(self, device: str) -> str:
        """デバイスの検証と設定"""
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            return 'cpu'
        return device
    
    @abstractmethod
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> Union[float, Dict[str, float]]:
        """メトリクス計算の抽象メソッド"""
        pass
    
    @abstractmethod
    def _setup_metric_specific_resources(self) -> None:
        """メトリクス固有のリソース初期化"""
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """テンソルを設定されたデバイスに移動"""
        return tensor.to(self.device)
```

### データクラスの活用

元のプロジェクトではデータクラスが多用されています：

```python
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class IndividualImageMetrics:
    """個別画像メトリクスのコンテナ"""
    # 基本メトリクス
    psnr_db: float
    ssim: float  
    mse: float
    mae: float
    # 高度なメトリクス
    lpips: Optional[float] = None
    ssim_improved: Optional[float] = None

@dataclass
class AllMetricsResults:
    """包括的評価結果のコンテナ"""
    individual_metrics: List[IndividualImageMetrics]
    fid_score: float
    statistics: Dict[str, Dict[str, float]]
    total_images: int
    evaluation_timestamp: str
    created_dataset_path: str
    original_dataset_path: str
    
    def get_metric_summary(self) -> str:
        """主要メトリクスのサマリーを取得"""
        psnr_mean = self.statistics.get('psnr', {}).get('mean', 0)
        ssim_mean = self.statistics.get('ssim', {}).get('mean', 0)
        return f"PSNR: {psnr_mean:.2f}dB, SSIM: {ssim_mean:.4f}"
```

### エラーハンドリングとロギング

```python
import logging
logger = logging.getLogger(__name__)

class LPIPSMetric(BaseMetric):
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        try:
            # 入力検証
            if img1.shape != img2.shape:
                raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
            
            # 範囲の正規化
            if img1.min() >= 0 and img1.max() <= 1:
                img1 = img1 * 2.0 - 1.0  # [0,1] → [-1,1]
            
            # 計算実行
            with torch.no_grad():
                result = self._compute_lpips(img1, img2)
                return result.item()
                
        except Exception as e:
            logger.error(f"LPIPS calculation failed: {e}")
            return None
```

### テスト実装パターン

元のプロジェクトのテストパターンに従います：

```python
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch

# プロジェクトパスの追加
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from torch_image_metrics.metrics.basic import PSNR
from tests.fixtures.test_helpers import print_test_header, print_test_result

class TestPSNR:
    """PSNRメトリクスのテストスイート"""
    
    @pytest.fixture
    def image_metrics(self):
        """テスト用ImageMetricsインスタンス"""
        return ImageMetrics(device='cpu')
    
    @pytest.fixture
    def sample_images(self):
        """テスト用サンプル画像"""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 32, 32)
        img2 = img1 + torch.randn_like(img1) * 0.1
        img2 = torch.clamp(img2, 0, 1)
        return img1, img2
    
    def test_psnr_calculation_normal(self, image_metrics, sample_images):
        """通常画像でのPSNR計算テスト"""
        print_test_header("PSNR Calculation - Normal Images")
        
        img1, img2 = sample_images
        psnr = image_metrics.calculate_psnr(img1, img2)
        
        assert isinstance(psnr, float)
        assert psnr > 0
        assert psnr < 100
        
        print_test_result("PSNR calculation", True, f"PSNR: {psnr:.2f} dB")
    
    def test_psnr_identical_images(self, image_metrics):
        """同一画像のPSNRテスト"""
        img = torch.ones(1, 3, 16, 16)
        psnr = image_metrics.calculate_psnr(img, img.clone())
        assert psnr == float('inf')
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_compatibility(self, device, sample_images):
        """デバイス互換性テスト"""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        img1, img2 = sample_images
        metrics = ImageMetrics(device=device)
        result = metrics.calculate_psnr(img1.to(device), img2.to(device))
        assert isinstance(result, float)
```

### 移行時の注意点

#### 1. インポートパスの修正
```python
# 元のコード
from generative_latent_optimization.metrics.image_metrics import ImageMetrics
from vae_toolkit import DeviceManager

# 新しいコード
from torch_image_metrics.metrics.basic import ImageMetrics
from torch_image_metrics.core.device_manager import DeviceManager
```

#### 2. vae-toolkit依存の削除
元のコードはvae-toolkitパッケージに依存していますが、必要な機能を抽出して独立させます：
- DeviceManager → torch_image_metrics.core.device_manager
- その他のユーティリティ → torch_image_metrics.utils

#### 3. テストヘルパーの移行
```python
# tests/fixtures/test_helpers.py を作成
def print_test_header(title: str):
    """テストヘッダー表示"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_test_result(name: str, passed: bool, details: str = ""):
    """テスト結果表示"""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {name}: {status} {details}")
```

### コード品質基準

#### 型ヒントの完全性
```python
from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np

def calculate_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    metrics: List[str] = ['psnr', 'ssim'],
    return_dict: bool = True
) -> Union[Dict[str, float], Tuple[float, ...]]:
    """型ヒントを完全に指定"""
    pass
```

#### docstringの記載
```python
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) を計算
    
    Args:
        img1: 参照画像テンソル [B, C, H, W] (範囲: [0, 1])
        img2: 比較画像テンソル [B, C, H, W] (範囲: [0, 1])
    
    Returns:
        PSNR値 (dB単位)
    
    Raises:
        ValueError: 画像の形状が一致しない場合
        
    Example:
        >>> img1 = torch.rand(1, 3, 256, 256)
        >>> img2 = torch.rand(1, 3, 256, 256)
        >>> psnr = calculate_psnr(img1, img2)
    """
    pass
```

### パフォーマンス最適化

#### バッチ処理の最適化
```python
def calculate_batch_metrics(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    metrics: List[str]
) -> Dict[str, torch.Tensor]:
    """バッチ全体を効率的に処理"""
    with torch.no_grad():  # 勾配計算を無効化
        results = {}
        
        # バッチ処理可能なメトリクスは一括処理
        if 'mse' in metrics:
            results['mse'] = F.mse_loss(batch1, batch2, reduction='none')
        
        # 個別処理が必要なメトリクスはループ
        if 'lpips' in metrics:
            lpips_values = []
            for i in range(batch1.shape[0]):
                lpips_values.append(
                    self.lpips_metric.calculate(batch1[i:i+1], batch2[i:i+1])
                )
            results['lpips'] = torch.tensor(lpips_values)
    
    return results
```

#### メモリ管理
```python
@torch.no_grad()
def process_large_dataset(
    dataset_path: str,
    batch_size: int = 32
) -> Dict[str, float]:
    """大規模データセットの効率的処理"""
    # メモリ効率的な処理
    torch.cuda.empty_cache()  # GPUキャッシュクリア
    
    results = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch_results = calculate_batch_metrics(batch)
        results.append(batch_results)
        
        # 定期的にメモリクリア
        if len(results) % 10 == 0:
            torch.cuda.empty_cache()
    
    return aggregate_results(results)
```

## 重要な参照情報

### 元のプロジェクトファイルマッピング

| 元ファイル (Generative-Latent-Optimization) | 新ファイル (torch-image-metrics) | 説明 |
|-------------------------------------------|----------------------------------|------|
| `src/.../metrics/image_metrics.py` | `src/.../metrics/basic.py` | PSNR, SSIM等の基本メトリクス |
| `src/.../metrics/individual_metrics.py` | `src/.../metrics/perceptual.py` | LPIPS等の知覚的メトリクス |
| `src/.../metrics/dataset_metrics.py` | `src/.../metrics/dataset.py` | FID等のデータセットメトリクス |
| `src/.../metrics/unified_calculator.py` | `src/.../calculator.py` | 統一計算器 |
| `src/.../evaluation/simple_evaluator.py` | `src/.../evaluator.py` | 評価器 |
| `src/.../core/base_classes.py` | `src/.../core/base_metric.py` | 基底クラス |

### 現在のタスク優先順位

1. **Phase 1 - 基本構造の確立** ⬅️ 現在ここ
   - [ ] プロジェクト構造の作成
   - [ ] 基底クラス（BaseMetric）の実装
   - [ ] データ構造（dataclass）の定義

2. **Phase 2 - コアメトリクスの移行**
   - [ ] PSNR, SSIM実装の移行
   - [ ] LPIPS実装の移行
   - [ ] FID実装の移行
   - [ ] 各メトリクスのテスト作成

3. **Phase 3 - 統一APIの実装**
   - [ ] Evaluatorクラスの実装
   - [ ] Calculatorクラスの実装
   - [ ] クイックAPI関数の実装

4. **Phase 4 - テストとドキュメント**
   - [ ] ユニットテストの完成
   - [ ] 統合テストの作成
   - [ ] ドキュメント作成

5. **Phase 5 - PyPI公開準備**
   - [ ] パッケージメタデータの完成
   - [ ] GitHub Actionsの設定
   - [ ] 初回リリース

### トラブルシューティング

#### よくある問題と解決方法

##### 1. CUDA関連のエラー
```bash
# エラー: CUDA out of memory
# 解決方法:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python your_script.py

# またはコード内で:
torch.cuda.empty_cache()
```

##### 2. 依存関係の問題
```bash
# lpipsパッケージが見つからない
uv add lpips

# pytorch-fidが見つからない  
uv add pytorch-fid
```

##### 3. テスト実行エラー
```bash
# パスの問題でインポートエラー
PYTHONPATH=. pytest tests/

# 特定のテストのデバッグ
pytest tests/unit/test_metrics/test_image_metrics.py -v --tb=short
```

##### 4. 型チェックエラー
```bash
# mypyの設定が必要な場合
cat > mypy.ini << EOF
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
EOF

mypy src/torch_image_metrics/
```

### 開発のヒント

#### 効率的なデバッグ
```python
# デバッグ用のプリント関数
def debug_tensor(tensor: torch.Tensor, name: str = "tensor"):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")
```

#### パフォーマンス測定
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.time()
    yield
    print(f"{name} took {time.time() - start:.3f} seconds")

# 使用例
with timer("PSNR calculation"):
    psnr = calculate_psnr(img1, img2)
```

### 関連リンクとリソース

- **元プロジェクト**: `~/ghq/github.com/mdipcit/Generative-Latent-Optimization`
- **参考ドキュメント**: `TORCH_IMAGE_METRICS_CLAUDE_CODE_REFERENCE.md`
- **PyTorch公式ドキュメント**: https://pytorch.org/docs/stable/
- **torchmetrics**: https://torchmetrics.readthedocs.io/
- **LPIPS**: https://github.com/richzhang/PerceptualSimilarity
- **pytorch-fid**: https://github.com/mseitzer/pytorch-fid

### コミットメッセージ規約

```bash
# フォーマット: <type>: <description>

feat: 新機能の追加
fix: バグ修正
refactor: コードのリファクタリング
test: テストの追加・修正
docs: ドキュメントの更新
perf: パフォーマンス改善
chore: ビルドプロセスやツールの変更

# 例:
git commit -m "feat: Add LPIPS metric implementation"
git commit -m "fix: Correct SSIM calculation for grayscale images"
git commit -m "test: Add unit tests for batch processing"
```

### 次のステップ

1. **プロジェクト構造の作成を開始**
   ```bash
   mkdir -p src/torch_image_metrics/{metrics,core,utils}
   touch src/torch_image_metrics/__init__.py
   ```

2. **基底クラスの実装**
   - 元プロジェクトから`BaseMetric`クラスを移行
   - vae-toolkit依存を削除

3. **最初のメトリクス実装**
   - PSNRとSSIMから開始（最もシンプル）
   - テストも同時に作成

---

このCLAUDE.mdファイルは、torch-image-metricsプロジェクトの開発を進める上で必要な全情報を含んでいます。
質問やサポートが必要な場合は、いつでもお尋ねください。