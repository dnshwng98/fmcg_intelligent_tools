# FMCG Intelligent Tools

Tools for performing intelligent data processing in FMCG domain

## How to Install the Dependencies
There are three ways to install the dependencies based on the attached requirements.txt file. You can choose one of the following:
1. **On Windows PowerShell**: Get-Content requirements.txt | ForEach-Object { poetry add $_ }
2. **Using pip**: pip install -r requirements.txt
3. **On Command Prompt**: FOR /F %i IN (requirements.txt) DO poetry add %i

## How to Run the Program
