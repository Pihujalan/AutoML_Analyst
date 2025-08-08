# 🧹 Repository Cleanup Plan for Public Release

## 📋 **Phase 1: Folder Structure Reorganization**

### **Current → New Structure:**
- `streamlit_app/` → `streamlit_app/`
- `data_processing/` → `data_processing/`
- `datasets/` → `datasets/` (optional)

## 📋 **Phase 2: Sensitive Information Removal**

### **API Keys to Remove/Replace:**
1. **ml/fullmodel.py** - Lines 36-40: Multiple API keys
2. **data_processing/dataCleaning.py** - Lines 65, 71-73: Multiple API keys
3. **mixtral_integration.py** - Line 235: API key placeholder
4. **creds.json** - Entire file (Google credentials)

### **Personal References to Remove:**
1. **data_processing/dataCleaning.py** - Line 69: "Mansi's" comment
2. **merged_pattern_analysis.py** - Line 204: "Pihu's Additional Functions"

## 📋 **Phase 3: Path Updates Required**

### **Files with Hardcoded Paths:**
1. **ml/fullmodel.py** - Multiple "data_processing/" and "streamlit_app/" references
2. **ml/rf.py** - "data_processing/" paths
3. **ml/rfauto.py** - "data_processing/" paths
4. **model_training.py** - "data_processing/" paths
5. **merged_pattern_analysis.py** - "data_processing/" paths
6. **test_predictions.py** - "data_processing/" paths
7. **streamlit_app/streamlit_interface.py** - "data_processing/" imports
8. **streamlit_app/data_processing/streamlit_interface_modified.py** - "data_processing/" imports

## 📋 **Phase 4: Configuration Files**

### **Files to Update:**
1. **streamlit_app/final_column_types.json** → **streamlit_app/final_column_types.json**
2. **streamlit_app/user_column_types.json** → **streamlit_app/user_column_types.json**
3. **data_processing/target_config.json** → **data_processing/target_config.json**
4. **data_processing/final_column_types.json** → **data_processing/final_column_types.json**

## 📋 **Phase 5: Documentation Updates**

### **Files to Create/Update:**
1. **README.md** - Comprehensive project documentation
2. **requirements.txt** - Dependencies list
3. **setup.py** - Installation script
4. **.env.example** - Environment variables template
5. **LICENSE** - Open source license

## 📋 **Phase 6: Code Quality**

### **Improvements:**
1. Remove debug prints and temporary files
2. Add proper error handling
3. Add docstrings and comments
4. Standardize code formatting
5. Remove unused imports

## 🚀 **Execution Order:**
1. Create backup of current state
2. Rename folders
3. Update all file paths
4. Remove sensitive information
5. Update imports and references
6. Test functionality
7. Update documentation
8. Final testing

## ⚠️ **Important Notes:**
- Keep backup of original structure
- Test each change incrementally
- Update .gitignore for new structure
- Remove any personal data from datasets
- Add proper licensing 