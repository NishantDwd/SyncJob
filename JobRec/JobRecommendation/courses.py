# Existing course lists remain the same...
ds_course = [['Machine Learning Crash Course by Google [Free]', 'https://developers.google.com/machine-learning/crash-course'],
             ['Machine Learning A-Z by Udemy','https://www.udemy.com/course/machinelearning/'],
             ['Machine Learning by Andrew NG','https://www.coursera.org/learn/machine-learning'],
             ['Data Scientist Master Program of Simplilearn (IBM)','https://www.simplilearn.com/big-data-and-analytics/senior-data-scientist-masters-program-training'],
             ['Data Science Foundations: Fundamentals by LinkedIn','https://www.linkedin.com/learning/data-science-foundations-fundamentals-5'],
             ['Data Scientist with Python','https://www.datacamp.com/tracks/data-scientist-with-python'],
             ['Programming for Data Science with Python','https://www.udacity.com/course/programming-for-data-science-nanodegree--nd104'],
             ['Programming for Data Science with R','https://www.udacity.com/course/programming-for-data-science-nanodegree-with-R--nd118'],
             ['Introduction to Data Science','https://www.coursera.org/specializations/introduction-data-science'],
             ['IBM Data Science Professional Certificate','https://www.coursera.org/professional-certificates/ibm-data-science']]

web_course = [['Django Crash Course','https://www.youtube.com/watch?v=e1IyzVyrLSU'],
              ['Python and Django Full Stack Web Developer Bootcamp','https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp'],
              ['React - The Complete Guide','https://www.udemy.com/course/react-the-complete-guide-incl-redux/'],
              ['Node.js, Express, MongoDB & More','https://www.udemy.com/course/nodejs-express-mongodb-bootcamp/'],
              ['The Complete Web Developer Course','https://www.udemy.com/course/the-complete-web-developer-course-2/'],
              ['Full-Stack Web Development with React Specialization','https://www.coursera.org/specializations/full-stack-react'],
              ['Modern React with Redux','https://www.udemy.com/course/react-redux/'],
              ['The Web Developer Bootcamp','https://www.udemy.com/course/the-web-developer-bootcamp/']]

android_course = [['Android Development for Beginners','https://www.udacity.com/course/android-development-for-beginners--ud837'],
                  ['Android App Development Specialization','https://www.coursera.org/specializations/android-app-development'],
                  ['The Complete Android Developer Course','https://www.udemy.com/course/complete-android-n-developer-course/'],
                  ['Developing Android Apps with Kotlin','https://www.udacity.com/course/developing-android-apps-with-kotlin--ud9012'],
                  ['Android Basics by Google','https://www.udacity.com/course/android-basics-nanodegree-by-google--nd803']]

ios_course = [['iOS Development for Creative Entrepreneurs','https://www.udacity.com/course/ios-development-for-creative-entrepreneurs--ud585'],
              ['iOS App Development with Swift Specialization','https://www.coursera.org/specializations/app-development'],
              ['iOS 13 & Swift 5 - The Complete iOS App Development Bootcamp','https://www.udemy.com/course/ios-13-app-development-bootcamp/'],
              ['Intro to iOS App Development with Swift','https://www.udacity.com/course/intro-to-ios-app-development-with-swift--ud585']]

uiux_course = [['Google UX Design Professional Certificate','https://www.coursera.org/professional-certificates/google-ux-design'],
               ['UI / UX Design Specialization','https://www.coursera.org/specializations/ui-ux-design'],
               ['The Complete App Design Course - UX, UI and Design Thinking','https://www.udemy.com/course/the-complete-app-design-course-ux-and-ui-design/'],
               ['User Experience Design Essentials','https://www.udemy.com/course/ui-ux-web-design-using-adobe-xd/'],
               ['Adobe XD Tutorial: User Experience Design Course','https://www.youtube.com/watch?v=68w2VwalD5w']]

# Enhanced and more specific keywords for better field detection
ds_keyword = [
    # Core ML/AI
    'machine learning', 'deep learning', 'artificial intelligence', 'neural network', 'ai', 'ml', 'dl',
    # Frameworks
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'sklearn', 'xgboost', 'lightgbm',
    # Data Science Tools
    'python', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
    # Techniques
    'data science', 'data analysis', 'data visualization', 'statistical modeling', 'statistical analysis',
    'predictive analysis', 'predictive modeling', 'time series', 'forecasting',
    # Specific ML Areas
    'nlp', 'natural language processing', 'computer vision', 'opencv', 'image processing',
    'recommendation system', 'clustering', 'classification', 'regression', 'ensemble',
    # Big Data
    'big data', 'apache spark', 'pyspark', 'hadoop', 'hive',
    # Other Tools
    'jupyter', 'anaconda', 'data mining', 'feature engineering', 'model deployment',
    'mlops', 'sagemaker', 'azure ml', 'google colab'
]

web_keyword = [
    # Frontend Frameworks - Major
    'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
    # Frontend - Core
    'javascript', 'typescript', 'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less',
    # Frontend - UI Libraries
    'bootstrap', 'tailwind', 'material-ui', 'mui', 'chakra ui', 'ant design',
    # Frontend - Tools
    'webpack', 'babel', 'npm', 'yarn', 'vite', 'redux', 'mobx', 'next.js', 'nuxt.js',
    # Backend Frameworks
    'node.js', 'nodejs', 'express', 'expressjs', 'django', 'flask', 'fastapi',
    'spring boot', 'spring', 'laravel', 'php', 'ruby on rails', 'asp.net', '.net core',
    # Databases
    'mongodb', 'mysql', 'postgresql', 'redis', 'elasticsearch', 'sql', 'nosql',
    # APIs & Architecture
    'rest api', 'restful', 'graphql', 'microservices', 'api', 'websocket',
    # Web Dev General
    'web development', 'full stack', 'frontend', 'backend', 'responsive design',
    'single page application', 'spa', 'server side rendering', 'ssr'
]

android_keyword = [
    # Core Android
    'android', 'android development', 'android studio', 'android sdk',
    # Languages
    'kotlin', 'java', 'android java',
    # Modern Android
    'jetpack compose', 'compose', 'android jetpack',
    # Architecture & Patterns
    'mvvm', 'mvc', 'mvp', 'clean architecture',
    # Android Libraries
    'retrofit', 'room', 'dagger', 'hilt', 'coroutines', 'rxjava', 'glide', 'picasso',
    # Android Components
    'activity', 'fragment', 'service', 'broadcast receiver', 'content provider',
    'livedata', 'viewmodel', 'lifecycle',
    # UI & Design
    'material design', 'xml', 'constraint layout', 'recyclerview',
    # Data & Storage
    'sqlite', 'shared preferences', 'datastore',
    # Backend Integration
    'firebase', 'fcm', 'google maps', 'rest api integration',
    # Tools
    'gradle', 'git', 'proguard', 'android ndk',
    # Cross-platform
    'flutter', 'react native', 'kivy'
]

ios_keyword = [
    # Core iOS
    'ios', 'ios development', 'iphone', 'ipad',
    # Languages
    'swift', 'objective-c', 'objective c', 'swiftui',
    # Frameworks
    'uikit', 'ui-kit', 'cocoa', 'cocoa touch', 'foundation',
    # Development Tools
    'xcode', 'instruments', 'testflight',
    # Architecture
    'mvvm', 'mvc', 'viper', 'coordinator pattern',
    # Core Features
    'core data', 'core animation', 'core graphics', 'core location',
    'av foundation', 'mapkit', 'cloudkit', 'healthkit', 'arkit',
    # UI & Layout
    'auto layout', 'storyboard', 'interface builder', 'size classes',
    # Networking
    'alamofire', 'urlsession', 'rest api', 'json parsing',
    # Data & Storage
    'sqlite', 'realm', 'user defaults', 'keychain',
    # Tools & Services
    'cocoapods', 'swift package manager', 'carthage',
    'firebase', 'push notifications', 'app store connect',
    # Modern iOS
    'combine', 'async await', 'swift concurrency', 'widgetkit'
]

uiux_keyword = [
    # Design Tools - Major
    'figma', 'adobe xd', 'sketch', 'invision', 'zeplin', 'framer',
    # Prototyping
    'prototyping', 'wireframe', 'wireframing', 'mockup', 'prototype',
    # Adobe Suite
    'adobe photoshop', 'photoshop', 'adobe illustrator', 'illustrator',
    'adobe after effects', 'after effects', 'adobe indesign', 'indesign',
    'adobe premiere pro', 'premiere pro',
    # UX Research & Methods
    'user research', 'user experience', 'ux research', 'usability testing',
    'user testing', 'a/b testing', 'user interview', 'persona', 'user persona',
    'user journey', 'journey map', 'empathy map',
    # Design Principles
    'ui', 'ux', 'user interface', 'interaction design', 'visual design',
    'information architecture', 'ia', 'design thinking', 'human-centered design',
    # Design Elements
    'typography', 'color theory', 'design system', 'style guide', 'brand identity',
    'responsive design', 'mobile design', 'web design',
    # Tools & Processes
    'balsamiq', 'axure', 'marvel', 'principle', 'flinto',
    'design sprint', 'agile ux', 'lean ux',
    # Modern Practices
    'accessibility', 'wcag', 'inclusive design', 'design ops',
    'component library', 'atomic design'
]