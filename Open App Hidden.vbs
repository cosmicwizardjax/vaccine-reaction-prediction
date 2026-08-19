Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = folder
pythonExe = folder & "\.venv\Scripts\python.exe"
appFile = folder & "\app.py"
shell.Run """" & pythonExe & """ -m streamlit run """ & appFile & """ --server.headless true --server.address 127.0.0.1""", 0, False
