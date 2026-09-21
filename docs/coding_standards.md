# 代码检查与交付规范

## 1. 修改范围

只修改本项目源码，不修改外部 perceptive grasp 仓库。保留现有策略和配置的行为；新增功能通过独立配置选择。真机运动验证与离线测试分开记录，不把离线通过写成实物抓取成功。

## 2. 提交前检查

运行 `scripts/check_source.py` 检查 Python 语法、JSON、Shell 语法，再运行相关回归测试；提交前执行 `git diff --check` 并审阅全部差异。仓库当前没有额外 pre-commit 插件或通用编译步骤。Python 依赖按 requirements 文件安装，板端二进制依赖另行检查。

至少运行 delivery、result_feedback、feedback_execution、joint_delivery、green_pipeline、green_persistent 测试。涉及定位、标定和轨迹算法时，追加对应模块的测试。不要在自动源码检查中发送机械臂动作。

## 3. K3 验证与发布

将待发布内容同步到 K3 独立验证目录，运行相关回归测试、`scripts/check_environment.sh` 和不带 `--execute` 的入口预览；核对文件哈希。已有控制进程运行时不得覆盖其文件。

检查不包含密码、令牌、虚拟环境、历史采样或录像，保留第三方许可证。代码验证完成后新建分支，再提交和推送。新现场需要重新标定；参考配置不得被误认成已验证的新安装配置。

## 4. 文档

README 写安装、配置及可复现命令，不写中间调试日记。文档中的命令、参数、返回值必须与代码一致；历史工具须明确标注适用策略。
