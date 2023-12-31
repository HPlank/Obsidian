总结了五个基本基准原则来指导 OWOD 基准构建，并为 OWOD 问题构建了一个健全和公平的基准。
1.标签完整性(Label Integrity)：测试时应该整合标签信息以进行公平评估。 
2.任务增量性(Task Increment)：将数据集分成多个任务，每个任务包含一组语义超类别的类别，以逐步增加任务的难度
3. 类别开放性(Class Openness)：测试集应该包含训练集中没有的类别，以模拟真实世界中的未知类别
4. 数据特异性(Data Specificity)：训练集、验证集和测试集之间不应该有交集，每个数据集内也不应该有重复数据。
5. 注释特异性(Annotation Specificity)：注释应该尽可能详细，以提高模型的泛化能力。
进一步引入了两个公平的指标，未知检测召回率 (UDR) 和未知检测精度 (UDP)，它们特定于 OWOD 任务，并从未知类的角度评估检测性能。

提出了一个简单有效的 OWOD 框架，该框架具有辅助提案 ADvisor (PAD) 模块和特定类别的驱逐分类器 (CEC)，以克服未知类和背景或已知类之间的困难区别。
非参数PAD可以帮助RPN在没有监督的情况下确认准确的未知建议，并进一步指导RPN的学习来区分未知建议和背景。
虽然CEC校准过度自信的激活边界，并通过特定于类的驱逐函数过滤掉令人困惑的预测，避免了检测模型过度自信将未知实例分类为已知类

OWOD 任务包含多个增量任务。在每个任务中，仅由已知类训练的 OWOD 模型应该正确检测已知类并将未知类识别为测试阶段的“未知”。然后人工注释者可以逐渐将标签分配给感兴趣的类别，下一个任务中的检测模型应该使用新添加的注释逐步学习这些类。

假设第 t 个任务中的已知集和未知集表示为 Kt 和 Ut。
在任务 t + 1 中，未知类的兴趣，不失一般性，称为 U(k)t ∈ Ut，在已知集合 Kt+1 = Kt ∪ U(k)t 中进行注释和添加。
当前未知集Ut+1 = Ut\U(k)t。这个过程一直持续到 UT = ∅

注释特异性：
在训练和验证阶段，只有已知类 K 的实例被分配标签 Y = [L, B]，其中 L ∈ K 表示类标签，而 B 是其对应的边界框的位置坐标。
在测试过程中，未知类的实例只会分配一个“未知”类标签及其对应的框坐标。
之前的工作 [11] 打破了这一原则，并错误地使用了包含未知类信息的完全注释验证集来训练基于能量的分类器，实现了误导性的收益。
标签完整性：该原则指出应该在测试期间集成标签信息以进行公平比较。

1)未知客观性:区分未知实例和背景[3]。
2)未知歧视:将未知实例与类似的已知类[3]区分开来。
3)增量冲突：现有已知类和新注释的已知类的学习之间的平衡。

UDR 表示未知类的准确定位率
UDP 展示了所有本地化未知实例的准确分类率。
将未知类的真阳性建议和假阴性建议分别称为 TPu、FNu 和 FN∗u 表示错误分类预测边界框召回的真实框的数量。因此，UDP 和 UDR 可以计算为：![[Pasted image 20231103102210.png]]


潜在的未知正建议表示为 P(u)+。
P(u)+ 可能包含正未知提议和无意义的背景，是混乱和不确定的。他们的客观性分数 S 也不可靠。
因此，我们需要引入顾问来确认 P(u)+ 的对象性，并选择更可靠的积极建议进行进一步检测。顾问产生可能的对象区域，称为̃P+
。̃P+ 可以由任何无监督的对象检测方法生成。
在本文中，我们简单地选择经典的非参数选择性搜索[21]，
它通过基于颜色、纹理、大小和形状兼容性的相似区域的分层分组来计算对象的这些可能区域。
因此，确认 P(u)+ 的第 i 个提案的对象性可以表示如下：
![[Pasted image 20231103152240.png]]
 | · |是集合的长度，而 I(·) 是 Kro5 颈部增量函数，当输入条件成立时等于 1，否则为 0。
 IOU（交集过并），也称为 Jaccard 索引，是比较两个任意形状之间的相似性最常用的度量 [4]。
 IOU越大，说明顾问和RPN都将该区域视为客观性较高的积极建议，这意味着顾问证实了RPN输出的这一建议。
 θ 是 IOU 分数过滤模棱两可的建议的阈值。我们将严格的值设置为 0.7，

我们将这些准确提案的类标签修改为“前景”，从而形成这些提案的伪标签。
然后，我们寻求它们的原始锚点 A(u)+ 并将它们从负锚点集合 A− 中移除到正锚集 A+。
最后，我们将新的锚集输入到RPN的类不可知分类器f中，计算其损失函数Lcls RP N如下:
![[Pasted image 20231103152856.png]]

提出了一个特定于类的驱逐分类器来驱逐来自预测已知类的混淆实例并重新分配它们的类预测。
我们充分利用已知类的注释信息来校准每个类的过度自信激活边界。
根据细化的激活边界，预测的边界框可以重新分配其类别预测并确定其预测类别。
可以通过每个预测框的驱逐函数 Φ 获得驱逐指标。
由于每个已知类可能具有其特定于类的激活区域，我们自适应地进行驱逐每个已知类内的操作。
如果所有类都确定通过它们对应的驱逐指标来驱逐这个框，我们将将其预测为“未知”类别。
否则，我们将对允许其存在的类的置信度进行排名，并选择最高的类作为其类标签。
我们首先通过驱逐函数 Φ 计算特定于类的驱逐指标。至于测试数据集的输出预测 ̄Y = [ ̄L, ̄B]，我们将 ̄Lc i 表示第 i 个样本是否属于第 c 个已知类别的概率。因此，我们可以计算第 i 个样本第 c 个类别的后续驱逐指标：
![[Pasted image 20231103160007.png]]
其中 ̃L = [ ̃L, ̃B] 是训练图像的输出预测
Y = [L, B] 是对应的真实标签。
M =∑| ̃B|j∑|Bc |k [I(IOU( ̃Bj , Bc k) > φ)] 计算满足 Kronecker delta 函数条件的样本数。
φ 是第 c 个类别可能的边界框的 IOU 阈值，
α 是调整驱逐程度的超参数。
然后，我们可以重新分配预测边界框的已知类预测：
![[Pasted image 20231103160312.png]]

