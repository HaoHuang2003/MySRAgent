这个目录用于存放仍然保持独立项目身份的第三方源码。

典型场景：

- 依赖库可以安装，但还在开发或调试阶段，需要本地 editable install。
- 依赖库是外部项目，我们希望保留源码供阅读、调试或打补丁。
- 依赖库拥有自己的包名、测试、构建配置或发布节奏。

使用方式是对具体项目执行 editable install，例如：

```bash
pip install -e ./third-party/nd2py
```

然后在 `sr_agent` 中仍然使用该库自己的顶层包名导入：

```python
import nd2py as nd
from nd2py.core.symbols import Number
```

不要把整个 `third-party/` 目录加入 `PYTHONPATH`，也不要通过 `third-party.<project>` 导入其中的代码。每个第三方项目都应该只有一个权威导入名，通常就是它安装后的顶层包名。

如果某个第三方库不能安装，并且我们决定把它作为 `sr_agent` 的内部实现细节维护，应将它放到 `src/sr_agent/_vendor/`，而不是放在这里。
