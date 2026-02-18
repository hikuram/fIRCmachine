# fIRCmachine

※ReadmeはCopilotにより生成

fIRCmachineは、反応経路（IRC）、遷移状態（TS）最適化、振動解析（VIB）などを自動化するPythonベースの計算化学ツールキットです。

## 特徴
- ASE, dmf, Sella, orb_models, PySCF, gpu4pyscf, tblite, cupy などの先端パッケージを活用
- 反応経路探索、TS最適化、振動解析を一括実行
- グローバル設定（default_config.py）による柔軟なワークフロー制御
- CLIからの簡単な実行

## インストール
### 依存パッケージ
- Python 3.8以降
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [dmf](https://github.com/hikuram/dmf)
- [Sella](https://sellegroup.github.io/sella/)
- [orb_models](https://github.com/hikuram/orb_models)
- [PySCF](https://pyscf.org/)
- [gpu4pyscf](https://github.com/hikuram/gpu4pyscf)
- [tblite](https://github.com/tblite/tblite)
- [cupy](https://cupy.dev/)
- numpy, pandas, scipy, seaborn
- [redox_benchmark](https://github.com/hikuram/redox_benchmark)（※下記参照）

`requirements.txt` を利用してインストールできます：

```bash
pip install -r requirements.txt
```

**注意:** `redox_benchmark` はPyPI未登録のため、以下で個別インストールしてください：
```bash
pip install --no-deps git+https://github.com/hikuram/redox_benchmark.git
```

## 使い方
- フルワークフロー（IRC, TS, VIB）
	```bash
	python fIRCmachine/fIRCmachine.py -d <出力ディレクトリ> -c <電荷>
	```
- IRCのみ
	```bash
	python fIRCmachine/pIRCmachine.py -d <出力ディレクトリ> -c <電荷>
	```
- 振動解析のみ
	```bash
	python fIRCmachine/sVIBmachine.py -d <出力ディレクトリ> -c <電荷>
	```

詳細な設定は `fIRCmachine/default_config.py` を編集、または各スクリプト先頭のコメントアウト行で上書き可能です。

## ライセンス
GPL-3.0 License

## 注意・謝辞
- 本ツールの一部コードは [ColabReaction](https://github.com/hikuram/ColabReaction) パッケージを参考・流用しています。
	- ColabReactionのライセンス・著作権表示に従ってください。
- 各依存パッケージのライセンスもご確認ください。

## 開発・コントリビュート


---

- 本READMEは依存パッケージやワークフローの追加に応じて随時更新してください。
