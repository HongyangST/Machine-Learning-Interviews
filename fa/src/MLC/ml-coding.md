# <a name="ml-coding"></a> ۲. کدنویسی ML و داده :robot:

مرحلهٔ کدنویسی ML در شرکت‌های مختلف شکل یکسانی ندارد. بعضی شرکت‌ها پیاده‌سازی الگوریتم‌های کلاسیک از صفر را می‌سنجند و بعضی دیگر مهارت عملی Python و PyTorch، مانند کار با tensor، پیش‌پردازش، معیارها و training loop را ارزیابی می‌کنند.

در هر دو قالب، مصاحبه‌کننده به درستی راه‌حل، پایداری عددی، کیفیت کد، مدیریت حالت‌های مرزی، پیچیدگی و توانایی شما در توضیح انتخاب‌های طراحی توجه می‌کند.

## روش استفاده از این فصل

- پاسخ لینک‌شده از هر سؤال را باز کنید. هر مسئلهٔ canonical یک فایل Python قابل اجرا در پوشهٔ [`problems/`](../../../src/MLC/problems/) دارد.
- با اجرای [`tests/test_problem_answers.py`](../../../src/MLC/tests/test_problem_answers.py) پیاده‌سازی‌ها را بررسی کنید و حالت‌های مرزی مهم را ببینید.
- [notebookهای قدیمی](../../../src/MLC/notebooks/) را به‌عنوان محتوای تکمیلی و اکتشافی بخوانید. بعضی از آن‌ها پیش از solutionهای canonical ساخته شده‌اند و ممکن است کامل نباشند.
- هر مسئلهٔ اولویت‌دار را ابتدا بدون نگاه‌کردن به مرجع پیاده کنید. سپس درستی، پیچیدگی و پوشش حالت‌های مرزی را مقایسه کنید.

## درجهٔ سختی و tag شرکت‌ها

هر سؤال کدنویسی یک درجهٔ سختی مشابه LeetCode دارد:

- **Easy:** معمولاً یک عملیات اصلی، state محدود و پیاده‌سازی متمرکز دارد و باید حدود ۱۵ دقیقه زمان ببرد.
- **Medium:** چند گام یا حالت مرزی غیرساده دارد. پیاده‌سازی قوی در مصاحبه معمولاً ۲۰ تا ۳۵ دقیقه زمان می‌برد.
- **Hard:** مسئله‌ای ۴۰ تا بیش از ۶۰ دقیقه‌ای با چند جزء وابسته، debugging پیشرفته یا trade-offهای سیستمی است.

درجهٔ سختی، ارزیابی تحریریه دربارهٔ prompt کامل همین مخزن است؛ نه ادعایی مبنی بر اینکه همهٔ منابع مسئله‌ای مشابه را یکسان ارزیابی می‌کنند. موارد دقیق با [TorchLeet](https://github.com/Exorust/TorchLeet) و [Deep-ML](https://www.deep-ml.com/problems) کالیبره شده‌اند.

برای promptهایی که تطابق مستقیم ندارند، همان rubric استفاده شده است. اگر منابع اختلاف داشته باشند، دامنه و حالت‌های مرزی خواسته‌شده در prompt این مخزن، درجهٔ نهایی را تعیین می‌کند.

tag شرکت فقط زمانی اضافه می‌شود که یک مرجع، همان مسئلهٔ پیاده‌سازی را به شرکت نسبت دهد. این tagها سیگنال تاریخی برای آماده‌سازی هستند و تضمین نمی‌کنند که شرکت اکنون همان سؤال را می‌پرسد. از منابع ثالث فقط metadata استفاده شده و صورت مسئله یا پاسخ آن‌ها کپی نشده است.

آزمون‌های مرجع را از ریشهٔ مخزن اجرا کنید:

```bash
uv run --with numpy python -m unittest discover -s src/MLC/tests -p "test_*.py"
```

## کدنویسی ML با PyTorch

مصاحبه‌های جدید ML ممکن است به‌جای پیاده‌سازی صرف الگوریتم‌ها از صفر، مهارت عملی PyTorch را بسنجند. راهنمای [مسئله‌های کدنویسی PyTorch](../../../src/MLC/pytorch-ml-coding.md) این موضوع‌ها را پوشش می‌دهد:

- یک مصاحبهٔ آزمایشی ۶۰ دقیقه‌ای دربارهٔ tensor، پیش‌پردازش، معیارها، loop آموزش/ارزیابی و debugging؛
- سؤال‌های utility در Python و clean code برای workflowهای ML؛
- عملیات tensor، dataset، batching، مدیریت device و autograd؛
- آموزش، optimization، mixed precision، checkpoint و reproducibility؛
- testing، debugging، deployment و سؤال‌های پیشرفتهٔ PyTorch؛
- چالش‌های کدنویسی همراه با پاسخ مرجع کوتاه.

## مسئله‌های اولویت‌دار کدنویسی ML

این مجموعه، سؤال‌های کلاسیک و همچنان رایج را با primitiveهای مدرنی ترکیب می‌کند که در مصاحبه‌های AI/ML بیشتر انتظار می‌روند.

| مسئله | سختی | tag شرکت | راه‌حل canonical | notebook تکمیلی | پوشش مورد انتظار از یک پاسخ قوی |
| --- | --- | --- | --- | --- | --- |
| softmax و cross-entropy با پایداری عددی | Easy | Apple، Meta، Google، Amazon | [پاسخ Python](../../../src/MLC/problems/classic_ml/softmax_cross_entropy.py) | — | کم‌کردن max، log-sum-exp، shape و اعتبارسنجی class index |
| رگرسیون خطی با gradient descent | Medium | — | [پاسخ Python](../../../src/MLC/problems/classic_ml/linear_regression.py) | [Linear regression](../../../src/MLC/notebooks/linear_regression_md.ipynb) | گرادیان vectorized، bias، مقیاس MSE و convergence |
| رگرسیون لجستیک با gradient descent | Hard | Google، Meta، Amazon | [پاسخ Python](../../../src/MLC/problems/classic_ml/logistic_regression.py) | [Logistic regression](../../../src/MLC/notebooks/logistic_regression_md.ipynb) | sigmoid پایدار، گرادیان binary cross-entropy و threshold |
| k-nearest neighbors | Medium | Uber، LinkedIn، Meta | [پاسخ Python](../../../src/MLC/problems/classic_ml/knn.py) | [k-NN](../../../src/MLC/notebooks/k_nearest_neighbors.ipynb) | فاصله‌های pairwise، انتخاب top-k، tie و پیچیدگی |
| خوشه‌بندی k-means | Medium | Uber، LinkedIn، Google، Amazon | [پاسخ Python](../../../src/MLC/problems/classic_ml/kmeans.py) | [k-means](../../../src/MLC/notebooks/k_means_2.ipynb) | initialization، assignment برداری، convergence و خوشهٔ خالی |
| split در decision tree | Medium | — | [پاسخ Python](../../../src/MLC/problems/classic_ml/decision_tree_split.py) | [Decision tree](../../../src/MLC/notebooks/decision_tree.ipynb) | thresholdهای نامزد، impurity وزن‌دار و شرط توقف |
| تحلیل مؤلفه‌های اصلی | Medium | — | [پاسخ Python](../../../src/MLC/problems/classic_ml/pca.py) | — | مرکزدهی، SVD/eigendecomposition، ترتیب مؤلفه‌ها و variance |
| convolution دوبعدی | Medium | — | [پاسخ Python](../../../src/MLC/problems/classic_ml/conv2d.py) | [Convolution](../../../src/MLC/notebooks/convolution.ipynb) | shape خروجی، stride و تفاوت cross-correlation با convolution |
| scaled dot-product attention | Medium | — | [پاسخ Python](../../../src/MLC/problems/language_models/scaled_dot_product_attention.py) | — | shapeهای Q/K/V، عبارت `1/sqrt(d_k)` و mask پیش از softmax پایدار |
| معیارهای binary و ROC-AUC | Medium | — | [پاسخ Python](../../../src/MLC/problems/classic_ml/binary_metrics_roc_auc.py) | — | مخرج صفر، نامتوازنی class، tie و تفسیر rank |
| reservoir sampling | Medium | — | [پاسخ Python](../../../src/MLC/problems/classic_ml/reservoir_sampling.py) | — | طول نامعلوم stream، احتمال یکنواخت و حافظهٔ `O(k)` |
| TF-IDF | Medium | — | [پاسخ Python](../../../src/MLC/problems/language_models/tfidf.py) | — | شمارش token، document frequency، smoothing و sparse scaling |

هر سؤال canonical اکنون فایل پاسخ مستقل خود را در [`problems/`](../../../src/MLC/problems/) دارد.

## الگوریتم‌های کلاسیک تکمیلی

این تمرین‌ها به‌ویژه وقتی با حوزهٔ تیم هدف هماهنگ باشند، برای follow-up مناسب‌اند:

- SVM خطی و hinge loss ([notebook](../../../src/MLC/notebooks/svm.ipynb)) — **سختی:** Medium
- قانون یادگیری perceptron ([notebook](../../../src/MLC/notebooks/perceptron.ipynb)) — **سختی:** Easy
- شبکهٔ عصبی feedforward و backpropagation ([notebook](../../../src/MLC/notebooks/feedforward.ipynb)) — **سختی:** Hard
- گسترش multiclass یا multilabel برای معیارها و lossها — **سختی:** Medium
- Naive Bayes برای طبقه‌بندی متن — **سختی:** Medium
- matrix factorization برای سیستم پیشنهاددهنده — **سختی:** Medium
- gradient boosting: loop آموزش را توضیح دهید و یک گام سادهٔ residual fitting را پیاده کنید — **سختی:** Hard

## سؤال‌های داده و sampling

- splitکردن train/validation/test بدون data leakage — **سختی:** Easy
- استانداردسازی ویژگی‌ها فقط با آمار مجموعهٔ آموزش — **سختی:** Easy
- مدیریت مقدارهای گمشده و categoryهای دیده‌نشده به‌شکل یکدست — **سختی:** Medium
- پیاده‌سازی sampling یکنواخت، stratified، وزن‌دار و reservoir — **سختی:** Medium
- ساخت mini-batch و padding دنباله‌های با طول متغیر — **سختی:** Medium
- تجمیع درست loss وزن‌دار بر اساس نمونه و معیارهای streaming — **سختی:** Medium

## هنگام مصاحبه چه چیزهایی را توضیح دهید

1. پیش از کدنویسی، shape ورودی، dtype، فرض‌ها و خروجی مورد انتظار را بیان کنید.
2. با یک baseline درست شروع کنید و سپس bottleneck را vectorize یا بهینه کنید.
3. پیچیدگی زمان و حافظه، از جمله هزینهٔ ماتریس‌های pairwise، را توضیح دهید.
4. پایداری عددی، ورودی خالی، tie، ویژگی ثابت و label نامعتبر را مدیریت کنید.
5. برای حالت عادی و دست‌کم یک حالت شکست یا مرزی، آزمون کوچک بنویسید.
6. توضیح دهید پیاده‌سازی برای مجموعه‌دادهٔ بزرگ، GPU، آموزش توزیع‌شده یا کتابخانهٔ production چه تغییری می‌کند.
