# <a name="breadth"></a> ۳. مبانی ML و گسترهٔ دانش

هدف این مصاحبه، ارزیابی دانش عمومی شما از مفاهیم ML از دو دیدگاه نظری و عملی است. برخلاف مصاحبهٔ ML depth، ساختار و دامنهٔ breadth interview میان مصاحبه‌کنندگان و داوطلبان مختلف معمولاً شباهت زیادی دارد.

بهترین روش آماده‌سازی، مرور یادداشت‌های دوره‌های ML همراه با منابع آنلاین باکیفیت است. منابع زیر برای من مفید بوده‌اند.

# ۱. دوره‌ها و منابع مرور

- [دورهٔ Machine Learning از Andrew Ng](https://www.coursera.org/learn/machine-learning)؛ [ویدئوهای درس در YouTube](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) نیز در دسترس‌اند.
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Deep Learning Nanodegree از Udacity](https://www.udacity.com/course/deep-learning-nanodegree--nd101) یا [Deep Learning Specialization در Coursera](https://www.coursera.org/specializations/deep-learning)

اگر مفاهیم را از قبل می‌دانید، منابع زیر برای مرور سریع مناسب‌اند:

- [ویدئوهای Machine Learning از StatQuest](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
- [آمار در StatQuest](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)، به‌ویژه برای نقش‌های Data Science
- [Machine Learning cheatsheets](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [فلش‌کارت‌های ML از Chris Albon](https://machinelearningflashcards.com/)

# ۲. موضوع‌های اصلی مبانی ML

مهم‌ترین موضوع‌هایی که باید پوشش دهید در ادامه آمده‌اند.

## ۱. مفاهیم کلاسیک ML

### دسته‌بندی الگوریتم‌های ML

- یادگیری نظارت‌شده، بدون نظارت و نیمه‌نظارت‌شده همراه با مثال
  - تفاوت classification، regression و clustering
- الگوریتم‌های parametric در برابر non-parametric
- الگوریتم‌های خطی در برابر غیرخطی

### یادگیری نظارت‌شده

- الگوریتم‌های خطی
  - رگرسیون خطی
    - least squares، residual، رگرسیون خطی در برابر multivariate
  - رگرسیون لجستیک
    - تابع هزینه، فرمول و کد؛ sigmoid و cross-entropy
  - Support Vector Machine
  - Linear Discriminant Analysis
- درخت تصمیم
  - logit و leaf
  - الگوریتم آموزش و شرط توقف
  - inference
  - pruning
- روش‌های ensemble
  - bagging و boosting همراه با مثال
  - Random Forest
  - AdaBoost، GBM و XGBoost
- مقایسهٔ الگوریتم‌ها
  - [TBD: LinkedIn lecture]
- بهینه‌سازی
  - gradient descent: مفهوم، فرمول و کد
  - SGD، Momentum، RMSprop و Adam
- تابع‌های زیان
  - logistic loss
  - cross-entropy؛ فرمول آن را نیز به خاطر داشته باشید
  - hinge loss برای SVM
- انتخاب ویژگی
  - اهمیت ویژگی
- ارزیابی و انتخاب مدل
  - معیارهای ارزیابی
    - TP، FP، TN و FN
    - confusion matrix
    - accuracy، precision، recall/sensitivity، specificity و F-score
      - برای مجموعه‌دادهٔ نامتوازن کدام معیار مناسب است؟
      - تفاوت precision و TPR و دلیل استفاده از precision
    - منحنی ROC: رابطهٔ TPR و FPR و انتخاب threshold
    - AUC برای مقایسهٔ مدل
    - تعمیم معیارها به طبقه‌بندی چندکلاسه
    - معیارهای مخصوص هر الگوریتم [TBD]
  - انتخاب مدل
    - cross-validation
      - k-fold cross-validation و انتخاب مقدار مناسب `k`

### یادگیری بدون نظارت

- خوشه‌بندی
  - مدل‌های centroid: خوشه‌بندی k-means
  - مدل‌های connectivity: خوشه‌بندی سلسله‌مراتبی
  - مدل‌های density: الگوریتم DBSCAN
- Gaussian Mixture Model
- Latent Semantic Analysis
- Hidden Markov Model یا HMM
  - فرایند Markov
  - transition probability و emission probability
  - الگوریتم Viterbi [پیشرفته]
- روش‌های کاهش بُعد
  - Principal Component Analysis یا PCA
  - Independent Component Analysis یا ICA
  - t-SNE

### بایاس و واریانس؛ کم‌برازش و بیش‌برازش

- روش‌های regularization
  - L1/L2 یا Lasso/Ridge

### sampling

- sampling یکنواخت
- reservoir sampling
- stratified sampling

### مدیریت داده

- دادهٔ گمشده
- دادهٔ نامتوازن
- تغییر توزیع داده

### پیچیدگی محاسباتی الگوریتم‌های ML

- [TBD]

## ۲. یادگیری عمیق

- شبکهٔ عصبی feedforward
  - درک عمیق سازوکار آن
  - [نمونه] تابع activation برای classهایی که mutually exclusive نیستند
- RNN
  - backpropagation through time یا BPTT
  - مشکل vanishing/exploding gradient
- LSTM
  - مشکل vanishing/exploding gradient
  - رفتار gradient
- dropout
  - روش اعمال dropout به LSTM
- مدل‌های seq2seq
- attention
  - self-attention
- معماری Transformer با جزئیات کامل
  - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- embeddingها، به‌ویژه word embedding

## ۳. ML آماری

### الگوریتم‌های Bayesian

- Naive Bayes
- برآورد Maximum a Posteriori یا MAP
- برآورد Maximum Likelihood یا ML

### معناداری آماری

- R-squared
- p-value

## ۴. موضوع‌های دیگر

- دادهٔ پرت
- معیارهای شباهت و عدم شباهت
  - Euclidean، Manhattan، Cosine و Mahalanobis [پیشرفته]

## ۵. مدل‌های پایه و مدل‌های زبانی بزرگ

> تغییر مهم مصاحبه‌های ML در سال ۲۰۲۶ این است که LLM و foundation model دیگر یک تخصص جانبی نیستند و بخشی از breadth مورد انتظار محسوب می‌شوند. برای طراحی سرتاسری سیستم هوش مصنوعی مولد، شامل RAG، عامل هوشمند و serving، [فصل ۴](./MLSD/ml-system-design.md) و مخزن [سیستم‌های هوش مصنوعی عامل‌محور](https://github.com/alirezadir/Agentic-AI-Systems.git) را ببینید.

### درون Transformer و LLM

- مرور self-attention: scaled dot-product attention، دلیل تقسیم score بر `√dₖ` برای جلوگیری از اشباع softmax و حفظ گرادیان پایدار، و multi-head attention
- گونه‌های attention و trade-off حافظه/throughput: **MHA → MQA → GQA → MLA**؛ MLA در DeepSeek استفاده شده است.
- positional encoding مطلق یا آموختنی، **RoPE**، ALiBi و گسترش context طولانی با position interpolation و YaRN
- **KV cache:** کلید و مقدار هر لایه را cache می‌کند تا برای هر token تازه به‌جای محاسبهٔ دوبارهٔ `O(n²)`، هزینه `O(n)` باشد. در context طولانی و batch بزرگ، بخش عمدهٔ حافظه را مصرف می‌کند.
- **FlashAttention:** attention مبتنی بر tiling و آگاه از IO در SRAM که آموزش با context طولانی را عملی‌تر کرد.
- اجزای block: pre-norm در برابر post-norm، **RMSNorm** و activationهای **SwiGLU / GeGLU**
- **Mixture-of-Experts یا MoE:** routing پراکندهٔ expert، متعادل‌سازی بار و تفاوت پارامتر فعال با کل پارامترها؛ مانند Mixtral و DeepSeek-V3
- tokenization با **BPE، byte-level BPE یا SentencePiece**، اندازهٔ vocabulary و context window
- scaling lawها، از جمله Chinchilla compute-optimal، و توانایی‌های emergent

### پایپ‌لاین آموزش؛ از pretraining تا post-training

1. **Pretraining:** پیش‌بینی self-supervised توکن بعدی روی corpus در مقیاس وب
2. **SFT یا instruction tuning:** آموزش نظارت‌شده با نمونه‌های `prompt → response`
3. **Preference alignment:** هماهنگ‌سازی با ترجیح انسانی و ایمنی از طریق RLHF یا روش‌های جایگزین
4. **Reasoning RL:** یادگیری تقویتی با پاداش قابل راستی‌آزمایی برای مدل‌های reasoning مانند o-series و DeepSeek-R1

### روش‌های SFT، RL و alignment در post-training سال ۲۰۲۶

یکی از موضوع‌های اصلی مصاحبه در سال ۲۰۲۶، انتخاب روش مناسب post-training بر اساس داده و compute است. stackهای جدید معمولاً چند مرحله را ترکیب می‌کنند، مانند `SFT → preference optimization → RL`.

| الگوریتم | دادهٔ لازم | مدل اضافه | هزینهٔ نسبی | زمان استفاده |
| --- | --- | --- | --- | --- |
| **SFT** | نمونهٔ دستور با `prompt→response` | ندارد | کم | آموزش قالب و instruction-following؛ معمولاً مرحلهٔ نخست |
| **RLHF (PPO)** | label ترجیح انسانی | reward model، critic/value و reference | زیاد | روش کلاسیک که تا حد زیادی جای خود را به روش‌های ساده‌تر داده است |
| **DPO** | جفت ترجیح؛ پاسخ انتخاب‌شده و ردشده | reference model | متوسط | alignment آفلاین پیش‌فرض؛ شبیه SFT و بدون RL loop |
| **SimPO** | جفت ترجیح | ندارد | متوسط رو به کم | DPO بدون reference model |
| **KTO** | رأی دودویی مثبت/منفی، بدون جفت | ندارد | کم | feedback ارزان یا noisy و بدون pair |
| **ORPO** | نمونه‌های دستور | ندارد | کم | ترکیب SFT و preference tuning در یک مرحله |
| **GRPO** | prompt؛ نمونه‌گیری گروهی پاسخ و group-relative advantage | critic ندارد | متوسط | RL بدون value network برای reasoning و ریاضی |
| **RLVR** | وظیفه با پاداش قابل راستی‌آزمایی | verifier خودکار | متوسط | کد، ریاضی یا tool use که صحت آن قابل بررسی است |

نکته‌های مهم: DPO، reward model و RL loop را به loss نظارت‌شده روی جفت‌ها تبدیل می‌کند. GRPO با reward نرمال‌شده در گروه، critic را حذف و نسبت به PPO حافظهٔ کمتری مصرف می‌کند. GRPO و RLVR در مدل‌های reasoning نقش مهمی دارند.

رتبه‌بندی الگوریتم‌ها به scale وابسته است؛ برای مثال online RL ممکن است نزدیک 1.5B بهتر باشد و SimPO نزدیک 7B. با DAPO برای پایدارسازی RL با chain-of-thought طولانی و RLAIF برای جایگزینی feedback انسانی با AI نیز آشنا باشید.

### تنظیم دقیق کم‌هزینه از نظر پارامتر (PEFT)

- trade-off میان full fine-tune و PEFT از نظر compute، storage و catastrophic forgetting
- **LoRA / QLoRA:** adapter کم‌رتبه؛ QLoRA مدل پایهٔ quantized را fine-tune می‌کند. adapter، prefix tuning و prompt tuning را نیز بشناسید.
- ابرپارامترهای مهم: rank و `α` در LoRA، learning rate، epoch و batch size

### بهینه‌سازی inference و model serving

- **quantization:** تفاوت PTQ و QAT؛ INT8، INT4 با GPTQ/AWQ، FP8 و quantization در KV cache. در context طولانی، KV می‌تواند از وزن‌ها بزرگ‌تر شود.
- **paged attention** و continuous batching در vLLM، همراه با prefix caching
- **speculative decoding** با draft و verify، distillation و MoE برای compute پراکنده
- parallelism از نوع tensor، pipeline و sequence؛ تفاوت مرحلهٔ prefill و decode

### decoding و in-context learning

- greedy، beam، temperature، top-k، top-p یا nucleus، repetition penalty و decoding مقید با JSON یا grammar
- zero-shot و few-shot، chain-of-thought، self-consistency و test-time compute یا inference-time scaling
- افزودن دانش با RAG، context طولانی یا fine-tuning و trade-offهای آن‌ها

### ارزیابی مدل مولد؛ «eval طراحی سیستم جدید است»

- محدودیت معیارهای کلاسیک در تولید open-ended، hallucination و calibration
- **RAG triad** در RAGAS: faithfulness، ارتباط پاسخ و ارتباط context؛ همراه با recall@k، MRR و nDCG برای retrieval
- **LLM-as-judge**، win-rate جفتی، Arena/Elo، golden set و regression test
- معیار agent: کیفیت انتخاب ابزار، موفقیت task یا step و پایبندی به trajectory
- benchmarkهایی مانند MMLU، GPQA و SWE-bench؛ safety، red-teaming و مقاومت در برابر jailbreak

## ۶. هوش مصنوعی مولد و چندوجهی

### مدل‌های پایهٔ چندوجهی

- نمایش مشترک برای متن، تصویر، صوت، ویدئو و action
- روش‌های fusion: dual encoder کنتراستی مانند **CLIP / SigLIP**، projector یا adapter به فضای token در LLM مانند **LLaVA**، cross-attention در Flamingo، early/late fusion و مدل native یا omni مانند GPT-4o و Gemini
- درک و تولید یکپارچه؛ tokenizer تصویر یا صوت مانند VQ-VAE برای تولید autoregressive

### مدل‌های Vision-Language یا VLM

- معماری: **vision encoder مانند ViT / SigLIP / DINOv2 → projector → LLM**
- وظیفه‌ها: VQA، captioning، OCR و درک سند، grounding/detection و درک chart یا UI
- نمونه‌ها: GPT-4o، Gemini، Claude، Qwen-VL، LLaVA، PaliGemma و InternVL
- آموزش: pretraining روی جفت تصویر-متن و visual instruction tuning

### مدل‌های Vision-Language-Action یا VLA

- گسترش VLM برای کنترل embodied یا robotic: ادراک، فهم دستور و تولید action، اغلب در یک forward pass
- نمایش action با **توکن‌های گسسته** در RT-2 و OpenVLA یا مقدار پیوسته با action head مبتنی بر diffusion/flow matching در π0
- نمونه‌ها:
  - **RT-2:** ساختهٔ Google DeepMind بر پایهٔ PaLI-X/PaLM-E؛ انتقال دانش وب و chain-of-thought به کنترل robot
  - **OpenVLA:** مدل 7B باز با DINOv2، SigLIP و Llama-2؛ آموزش‌دیده با 970k demo واقعی و بهتر از RT-2-X با حدود ۷ برابر پارامتر کمتر
  - **π0 (Pi-Zero):** ترکیب PaliGemma VLM با expert مبتنی بر flow matching برای کنترل دقیق نزدیک 50 Hz
- کاربردها: manipulation ربات، humanoid و policy عمومی؛ داده از teleoperation demo و Open X-Embodiment

### تولید diffusion در برابر autoregressive

| ویژگی | **Autoregressive (AR)** | **Diffusion** |
| --- | --- | --- |
| روش | پیش‌بینی ترتیبی token بعدی | denoising تکرارشونده از noise |
| likelihood | دقیق | variational یا score-based |
| نقطهٔ قوت | دنبالهٔ گسسته، طول متغیر و reasoning | دادهٔ پیوستهٔ پربُعد مانند تصویر، ویدئو و صوت با fidelity زیاد |
| سرعت | یک forward برای هر token؛ KV cache کمک می‌کند | چندین گام denoising؛ با distillation، consistency یا flow matching کاهش می‌یابد |
| نمونه | متن GPT، token تصویر در Parti/VAR و صوت در AudioLM | Stable Diffusion و Imagen، ویدئو با Sora/Veo/DiT، Stable Audio و action ربات در π0 |

- **جزئیات diffusion:** فرایند forward/reverse در DDPM، پیش‌بینی score یا noise، latent diffusion، DiT برای ویدئو، classifier-free guidance، sampler سریع DDIM و flow matching/rectified flow در SD3 و π0
- **جزئیات AR:** تبدیل modality به token و پیش‌بینی token بعدی؛ مناسب مدل چندوجهی یکپارچه، image AR مانند VAR و صوت مانند AudioLM/MusicGen
- روندهای تازه: text diffusion یا masked diffusion LM، مدل consistency یا few-step و stack ترکیبی AR + diffusion

# ۳. نمونه سؤال‌های مبانی ML

- یادگیری ماشین چیست و چه تفاوتی با برنامه‌نویسی سنتی دارد؟
- روش‌های مختلف یادگیری ماشین کدام‌اند؟
- یادگیری نظارت‌شده، بدون نظارت و نیمه‌نظارت‌شده چه تفاوتی دارند؟
- مراحل ساخت یک مدل یادگیری ماشین چیست؟
- trade-off بایاس و واریانس را توضیح دهید.
- بیش‌برازش چیست و چگونه از آن جلوگیری می‌کنید؟
- چرا و چگونه داده را به train، validation و test تقسیم می‌کنید؟
- cross-validation چیست و چرا اهمیت دارد؟
- regularization و انواع L1 و L2 را توضیح دهید.
- دادهٔ گمشده یا خراب را در مجموعه‌داده چگونه مدیریت می‌کنید؟
- درخت تصمیم و رگرسیون لجستیک چگونه کار می‌کنند؟
- الگوریتم KNN را توضیح دهید و آن را با k-means مقایسه کنید.
- الگوریتم‌های مبتنی بر درخت مانند Random Forest و GBDT را توضیح دهید.
- gradient descent و SVM چگونه کار می‌کنند؟ Kernel SVM چیست؟
- شبکهٔ عصبی و backpropagation چگونه کار می‌کنند؟
- یادگیری عمیق چه تفاوتی با ML کلاسیک دارد؟
- CNN چیست و چگونه کار می‌کند؟
- یادگیری انتقالی چیست و در عمل چگونه استفاده می‌شود؟
- [۴۵ سؤال مصاحبهٔ ML](https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-interview-questions)

### نمونه سؤال‌های LLM، هوش مصنوعی مولد و multimodal در سال ۲۰۲۶

- یک block ترنسفورمر را مرحله‌به‌مرحله توضیح دهید. چرا scoreهای attention بر `√dₖ` تقسیم می‌شوند؟
- KV cache چیست، چرا برای inference مهم است و حافظهٔ آن چگونه scale می‌شود؟
- MHA، MQA و GQA را مقایسه کنید. چرا Llama از GQA استفاده کرد؟
- RoPE چیست و چرا rotary embedding بر positional embedding آموختنی ترجیح داده می‌شود؟
- چه زمانی RAG، fine-tuning یا context طولانی را انتخاب می‌کنید؟
- SFT، DPO، GRPO و RLVR را از نظر داده، مدل اضافه و کاربرد مقایسه کنید.
- چرا GRPO شبکهٔ critic را حذف می‌کند و advantage را چگونه تخمین می‌زند؟
- LoRA و QLoRA چیستند و چه زمانی PEFT را به full fine-tuning ترجیح می‌دهید؟
- trade-off دقت و latency در quantization با INT8/INT4، FP8 و KV-cache quantization چیست؟
- speculative decoding چگونه سرعت تولید را افزایش می‌دهد؟
- سیستم LLM یا RAG را چگونه ارزیابی می‌کنید؟ RAG triad و LLM-as-judge چیستند؟
- VLM چگونه vision encoder را با LLM ترکیب می‌کند؟
- VLA چیست؟ توکن action گسسته را با action head مبتنی بر flow matching مقایسه کنید.
- diffusion و autoregressive generation چه تفاوتی دارند و برای متن، تصویر، ویدئو یا صوت کجا استفاده می‌شوند؟
- classifier-free guidance و flow matching یا rectified flow چیست؟
- مدل Mixture-of-Experts چیست و پارامتر فعال با کل پارامتر چه تفاوتی دارد؟
