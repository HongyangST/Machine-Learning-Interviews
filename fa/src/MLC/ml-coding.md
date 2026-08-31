# <a name="ml-coding"></a> ۲. کدنویسی ML و داده :robot:

این صفحه فهرست مرجع تمرین‌های کدنویسی برای مصاحبه‌های ML است. هر بار روی یک دسته تمرکز کنید: ابتدا صورت مسئله را بدون دیدن پاسخ پیاده کنید، نتیجه را با پاسخ لینک‌شده مقایسه کنید، حالت‌های مرزی را بیازمایید و trade-offها را با صدای بلند توضیح دهید.

این فهرست عمداً بر اساس مهارت مصاحبه دسته‌بندی شده است، نه framework:

1. **ML کلاسیک** — الگوریتم‌ها، آماده‌سازی داده، معیارها و مبانی آموزش.
2. **مدل‌های زبانی (LM)** — tokenization، attention، decoding و batch‌بندی دنباله‌ها.
3. **هوش مصنوعی مولد** — سازگارکردن مدل، retrieval، یادگیری ترجیح و primitiveهای تولید.
4. **کدنویسی هوش مصنوعی عامل‌محور** — ابزارها، orchestration، حافظه، retry، مجوزها و پایان‌دادن به اجرا.

تمام ردیف‌های فهرست پاسخ دارند. پاسخ‌های Python عمداً کوتاه و مناسب مصاحبه‌اند و به‌صورت مستقل اجرا می‌شوند؛ برای چند تمرین تصویریِ شناخته‌شده، notebook موجود به‌عنوان پاسخ استفاده شده است.

## روش استفاده از این فصل

1. یک سؤال را انتخاب کنید و ورودی، خروجی، shapeها و فرض‌های آن را با بیان خودتان توضیح دهید.
2. بدون بازکردن پاسخ، یک baseline درست پیاده کنید.
3. یک حالت عادی و دست‌کم یک حالت مرزی یا شکست را آزمایش کنید.
4. پیاده‌سازی را با پاسخ لینک‌شده مقایسه کنید و سپس پیچیدگی و trade-offهای محیط production را توضیح دهید.
5. سؤال‌های **Easy** را در ۱۵ دقیقه، **Medium** را در ۲۰ تا ۳۵ دقیقه و **Hard** را در ۴۰ تا بیش از ۶۰ دقیقه دوباره حل کنید.

همهٔ آزمون‌های پاسخ‌های Python را از ریشهٔ مخزن اجرا کنید:

```bash
uv run --with numpy python -m unittest discover -s src/MLC/tests -p "test_*.py"
```

رنگ درجهٔ سختی: ![Easy](../../../src/MLC/assets/difficulty-easy.svg) ![Medium](../../../src/MLC/assets/difficulty-medium.svg) ![Hard](../../../src/MLC/assets/difficulty-hard.svg)

## مسئله‌های اولویت‌دار کدنویسی ML

### ML کلاسیک

این سؤال‌ها الگوهای اصلی پیاده‌سازی در مصاحبه‌های عمومی ML، علوم کاربردی و نقش‌های داده‌محور را پوشش می‌دهند.

| مسئله | سختی | tagها | tag شرکت | پاسخ | تمرکز مصاحبه |
| --- | --- | --- | --- | --- | --- |
| softmax و cross-entropy با پایداری عددی | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | `NumPy`، احتمال، پایداری عددی | Apple، Meta، Google، Amazon | [Python](../../../src/MLC/problems/classic_ml/softmax_cross_entropy.py) | کم‌کردن max، log-sum-exp، shape و اعتبارسنجی class index |
| رگرسیون خطی با gradient descent | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | رگرسیون، optimization، vectorization | — | [Python](../../../src/MLC/problems/classic_ml/linear_regression.py) · [Notebook](../../../src/MLC/notebooks/linear_regression_md.ipynb) | گرادیان‌های MSE، bias، مقیاس ویژگی و convergence |
| رگرسیون لجستیک با gradient descent | ![Hard](../../../src/MLC/assets/difficulty-hard.svg) | طبقه‌بندی، optimization، پایداری عددی | Google، Meta، Amazon | [Python](../../../src/MLC/problems/classic_ml/logistic_regression.py) · [Notebook](../../../src/MLC/notebooks/logistic_regression_md.ipynb) | sigmoid پایدار، target دودویی، گرادیان و threshold تصمیم |
| k-nearest neighbors | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | فاصله، vectorization، طبقه‌بندی | Uber، LinkedIn، Meta | [Python](../../../src/MLC/problems/classic_ml/knn.py) · [Notebook](../../../src/MLC/notebooks/k_nearest_neighbors.ipynb) | فاصله‌های pairwise، انتخاب top-k، tie و هزینهٔ زمان و حافظه |
| خوشه‌بندی k-means | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | خوشه‌بندی، یادگیری بدون ناظر، convergence | Uber، LinkedIn، Google، Amazon | [Python](../../../src/MLC/problems/classic_ml/kmeans.py) · [Notebook](../../../src/MLC/notebooks/k_means_2.ipynb) | initialization، گام‌های assignment/update و خوشهٔ خالی |
| split در decision tree | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | درخت، Gini impurity، جست‌وجو | — | [Python](../../../src/MLC/problems/classic_ml/decision_tree_split.py) · [Notebook](../../../src/MLC/notebooks/decision_tree.ipynb) | thresholdهای نامزد، impurity وزن‌دار و شرط توقف |
| تحلیل مؤلفه‌های اصلی | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | جبر خطی، کاهش بُعد، SVD | — | [Python](../../../src/MLC/problems/classic_ml/pca.py) | مرکزدهی، ترتیب مؤلفه‌ها و variance توضیح‌داده‌شده |
| convolution دوبعدی | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | یادگیری عمیق، بینایی ماشین، shapeهای tensor | — | [Python](../../../src/MLC/problems/classic_ml/conv2d.py) · [Notebook](../../../src/MLC/notebooks/convolution.ipynb) | shape خروجی، stride و تفاوت cross-correlation با convolution |
| معیارهای دودویی و ROC-AUC | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | ارزیابی، رتبه‌بندی، نامتوازنی کلاس | — | [Python](../../../src/MLC/problems/classic_ml/binary_metrics_roc_auc.py) | مخرج صفر، scoreهای مساوی و تفسیر rank |
| reservoir sampling | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | stream، احتمال، sampling | — | [Python](../../../src/MLC/problems/classic_ml/reservoir_sampling.py) | طول نامعلوم stream، احتمال یکنواخت ورود و حافظهٔ `O(k)` |
| SVM خطی و hinge loss | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | روش‌های margin، optimization، طبقه‌بندی | — | [Notebook](../../../src/MLC/notebooks/svm.ipynb) | hinge loss، نقض margin و regularization |
| قانون یادگیری perceptron | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | یادگیری آنلاین، طبقه‌بندی خطی | — | [Notebook](../../../src/MLC/notebooks/perceptron.ipynb) | شرط update، فرض‌های convergence و مرز تصمیم |
| شبکهٔ عصبی feedforward و backpropagation | ![Hard](../../../src/MLC/assets/difficulty-hard.svg) | شبکهٔ عصبی، chain rule، گرادیان | — | [Notebook](../../../src/MLC/notebooks/feedforward.ipynb) | cache مرحلهٔ forward، جریان گرادیان، update پارامترها و بررسی shape |
| معیارهای multiclass و میانگین‌گیری macro | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | ارزیابی، confusion matrix، میانگین‌گیری | — | [Python](../../../src/MLC/problems/classic_ml/multiclass_metrics.py) | شمارش هر کلاس، macro در برابر micro و کلاس‌های غایب |
| Multinomial Naive Bayes برای متن | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | NLP، احتمال، smoothing | — | [Python](../../../src/MLC/problems/classic_ml/multinomial_naive_bayes.py) | فضای log، Laplace smoothing، tokenهای دیده‌نشده و فرض استقلال |
| matrix factorization برای سیستم پیشنهاددهنده | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | پیشنهاد، embedding، SGD | — | [Python](../../../src/MLC/problems/classic_ml/matrix_factorization.py) | درایه‌های مشاهده‌شده، regularization، رتبهٔ latent و cold start |
| یک گام residual در gradient boosting | ![Hard](../../../src/MLC/assets/difficulty-hard.svg) | ensemble، برازش residual، درخت رگرسیون | — | [Python](../../../src/MLC/problems/classic_ml/gradient_boosting_step.py) | گرادیان منفی، weak learner و shrinkage نرخ یادگیری |
| جداسازی train/validation/test بدون leakage | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | جداسازی داده، reproducibility، stratification | — | [Python](../../../src/MLC/problems/classic_ml/dataset_split.py) | indexهای جدا، تعادل کلاس و fit فقط روی train |
| استانداردسازی ویژگی با آمار مجموعهٔ آموزش | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | پیش‌پردازش، leakage، پایداری عددی | — | [Python](../../../src/MLC/problems/classic_ml/feature_standardization.py) | جداسازی fit/transform، ویژگی ثابت و اعتبارسنجی shape |
| مقدارهای گمشده و categoryهای دیده‌نشده | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | پیش‌پردازش، imputation، encoding ویژگی دسته‌ای | — | [Python](../../../src/MLC/problems/classic_ml/categorical_preprocessing.py) | state فقط از train، bucket ناشناخته و ستون کاملاً خالی |
| sampling یکنواخت، وزن‌دار و stratified | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | sampling، نامتوازنی، reproducibility | — | [Python](../../../src/MLC/problems/classic_ml/sampling_strategies.py) | replacement، وزن‌های نرمال‌شده و تضمین هر کلاس |
| loss وزن‌دار و معیارهای streaming | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | ارزیابی، streaming، تجمیع توزیع‌شده | — | [Python](../../../src/MLC/problems/classic_ml/streaming_weighted_mean.py) | وزن‌دهی با تعداد نمونه، state قابل ادغام و state خالی |

### مدل‌های زبانی (LM)

این سؤال‌ها از ویژگی‌های متنی تا سازوکارهای token، attention و decoding در مدل‌های زبانی مدرن پیش می‌روند.

| مسئله | سختی | tagها | tag شرکت | پاسخ | تمرکز مصاحبه |
| --- | --- | --- | --- | --- | --- |
| TF-IDF | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | NLP، ویژگی sparse، بازیابی اطلاعات | — | [Python](../../../src/MLC/problems/language_models/tfidf.py) | شمارش token، document frequency، smoothing و ترتیب واژگان |
| scaled dot-product attention | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | attention، transformer، masking | — | [Python](../../../src/MLC/problems/language_models/scaled_dot_product_attention.py) | shapeهای Q/K/V، عبارت `1/sqrt(d_k)` و mask پیش از softmax پایدار |
| یادگیری و اعمال byte-pair encoding | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | tokenization، BPE، واژگان | — | [Python](../../../src/MLC/problems/language_models/byte_pair_encoding.py) | شمارش pair، merge قطعی و encoding واژهٔ دیده‌نشده |
| ساخت causal attention mask | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | transformer، masking، decoding خودبازگشتی | — | [Python](../../../src/MLC/problems/language_models/causal_attention_mask.py) | معنای query/key، padding و قرارداد mask بولی |
| positional encoding سینوسی | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | transformer، موقعیت، vectorization | — | [Python](../../../src/MLC/problems/language_models/sinusoidal_position_encoding.py) | کانال‌های متناوب sine/cosine، عرض فرد و shape |
| sampling با temperature، top-k و top-p | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | decoding، احتمال، تولید | Anthropic، OpenAI، DeepMind | [Python](../../../src/MLC/problems/language_models/top_k_top_p_sampling.py) | ترتیب فیلتر، نرمال‌سازی دوباره و آزمون قطعی |
| KV cache فقط-افزودنی | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | inference، attention، latency، حافظه | Anthropic، OpenAI، Meta، Perplexity | [Python](../../../src/MLC/problems/language_models/kv_cache.py) | shape حافظه، ظرفیت، prefill در برابر decode و رشد حافظه |
| padding برای batchهای token با طول متغیر | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | batching، padding، mask | — | [Python](../../../src/MLC/problems/language_models/padded_batches.py) | شناسهٔ padding، mask اعتبار، دنبالهٔ خالی و shape batch |

### هوش مصنوعی مولد

این سؤال‌ها پیاده‌سازی‌های فشرده‌ای از اجزای رایج fine-tuning، retrieval، alignment و تولید هستند.

| مسئله | سختی | tagها | tag شرکت | پاسخ | تمرکز مصاحبه |
| --- | --- | --- | --- | --- | --- |
| update روش LoRA برای یک لایهٔ خطی | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | fine-tuning، low rank، جبر خطی | Meta، Google، Anthropic، OpenAI، Databricks | [Python](../../../src/MLC/problems/genai/lora_linear.py) | shape ماتریس، مقیاس `alpha/r`، صرفه‌جویی پارامتر و merge |
| contrastive loss متقارن | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | embedding، InfoNCE، چندوجهی | OpenAI، Anthropic، DeepMind، Midjourney | [Python](../../../src/MLC/problems/genai/contrastive_loss.py) | نرمال‌سازی، temperature، نمونه‌های منفی درون batch و دو جهت |
| retrieval دقیق cosine top-k برای RAG | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | RAG، embedding، retrieval، رتبه‌بندی | — | [Python](../../../src/MLC/problems/genai/rag_retrieval.py) | نرمال‌سازی، ترتیب top-k، بردار صفر و trade-off روش ANN در production |
| loss روش Direct Preference Optimization | ![Hard](../../../src/MLC/assets/difficulty-hard.svg) | alignment، یادگیری ترجیح، fine-tuning | Anthropic، OpenAI، DeepMind، Meta | [Python](../../../src/MLC/problems/genai/dpo_loss.py) | فاصلهٔ log-probability در policy/reference، beta و log-sigmoid پایدار |
| ترکیب classifier-free guidance | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | diffusion، guidance، تولید | — | [Python](../../../src/MLC/problems/genai/classifier_free_guidance.py) | پیش‌بینی شرطی/بدون شرط و trade-off مقیاس guidance |

### کدنویسی هوش مصنوعی عامل‌محور

این سؤال‌ها control plane قطعی پیرامون مدل را می‌سنجند. پاسخ‌های مرجع، مجوزها، budgetها، retry و شرط پایان را در کد نگه می‌دارند و صرفاً به متن prompt متکی نیستند.

| مسئله | سختی | tagها | tag شرکت | پاسخ | تمرکز مصاحبه |
| --- | --- | --- | --- | --- | --- |
| اعتبارسنجی آرگومان‌های JSON ابزار | ![Easy](../../../src/MLC/assets/difficulty-easy.svg) | ابزار، schema، اعتبارسنجی | — | [Python](../../../src/MLC/problems/agentic_ai/tool_argument_validation.py) | فیلدهای الزامی، type، enum و آرگومان اضافه |
| registry ابزار با آگاهی از مجوز | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | ابزار، مجوز، function calling | — | [Python](../../../src/MLC/problems/agentic_ai/tool_registry.py) | registration، تطبیق signature و مرز تأیید side effect |
| مرز planner/executor | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | planning، orchestration، tracing | — | [Python](../../../src/MLC/problems/agentic_ai/planner_executor.py) | plan محدود، نتیجهٔ ساخت‌یافته و شکست جزئی |
| retry همراه با idempotency | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | قابلیت اطمینان، retry، side effect | — | [Python](../../../src/MLC/problems/agentic_ai/retry_with_idempotency.py) | خطای موقت، exponential backoff و جلوگیری از اجرای تکراری |
| حافظهٔ مکالمه با پنجرهٔ لغزان | ![Medium](../../../src/MLC/assets/difficulty-medium.svg) | حافظه، context window، budget توکن | — | [Python](../../../src/MLC/problems/agentic_ai/sliding_window_memory.py) | دستورهای ثابت، تاریخچهٔ اخیر، eviction و trade-off خلاصه‌سازی |
| loop محدود استفادهٔ عامل از ابزار | ![Hard](../../../src/MLC/assets/difficulty-hard.svg) | عامل، ابزار، پایان، ایمنی | — | [Python](../../../src/MLC/problems/agentic_ai/bounded_agent_loop.py) | budget گام، تشخیص فراخوانی تکراری، ابزار ناشناخته و traceپذیری |

## تمرین‌های تکمیلی

- [مسئله‌های کدنویسی PyTorch](../../../src/MLC/pytorch-ml-coding.md) بانک تمرین تکمیلی و وابسته به framework است؛ چهار جدول بالا فهرست مرجع سؤال‌ها هستند.
- [notebookهای قدیمی](../../../src/MLC/notebooks/) برای تمرین‌های تصویری و اکتشافی در دسترس‌اند. جدول‌های مرجع فقط به notebookهایی لینک می‌دهند که اکنون پاسخ یا مکمل پاسخ‌اند.
- [آمادگی مصاحبه برای سیستم‌های هوش مصنوعی عامل‌محور](https://github.com/alirezadir/Agentic-AI-Systems/tree/main/06_interview_prep) لایهٔ طراحی سیستم را پوشش می‌دهد که مکمل تمرین‌های کدنویسی عامل‌محور این فصل است.

## سیاست درجهٔ سختی و tag شرکت‌ها

- **Easy:** یک عملیات متمرکز، معمولاً در حدود ۱۵ دقیقه.
- **Medium:** چند گام یا حالت مرزی معنادار، معمولاً ۲۰ تا ۳۵ دقیقه.
- **Hard:** اجزای وابسته، debugging پیشرفته یا trade-offهای سیستمی، معمولاً ۴۰ تا بیش از ۶۰ دقیقه.

درجهٔ سختی، ارزیابی تحریریه دربارهٔ prompt کامل است. موارد دقیق با [TorchLeet](https://github.com/Exorust/TorchLeet) و [Deep-ML](https://www.deep-ml.com/problems) کالیبره شده‌اند و برای سؤال‌های بدون تطابق، همان rubric مبتنی بر دامنه به کار می‌رود. tag شرکت فقط زمانی نمایش داده می‌شود که یک مرجع، شرکت را به همان مسئلهٔ پیاده‌سازی نسبت داده باشد. این tagها سیگنال تاریخی برای آماده‌سازی‌اند، نه ادعایی دربارهٔ روند فعلی مصاحبه. این مخزن فقط از metadata استفاده می‌کند و صورت مسئله یا پاسخ منابع ثالث را کپی نمی‌کند.

## هنگام مصاحبه چه چیزهایی را توضیح دهید

1. پیش از کدنویسی، shapeها، dtypeها، فرض‌ها و خروجی مورد انتظار را بیان کنید.
2. با یک baseline درست شروع کنید و سپس bottleneck را بهینه کنید.
3. پیچیدگی زمان و حافظه، از جمله ماتریس‌های میانی بزرگ و state مدل را توضیح دهید.
4. پایداری عددی، ورودی خالی، label نامعتبر، tie و مرز مجوزها را مدیریت کنید.
5. برای حالت عادی و دست‌کم یک حالت شکست یا مرزی آزمون بنویسید.
6. توضیح دهید پاسخ برای مجموعه‌دادهٔ بزرگ، GPU، اجرای توزیع‌شده یا سرویس production چه تغییری می‌کند.
