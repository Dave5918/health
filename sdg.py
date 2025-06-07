import streamlit as st
st.markdown("<h3 style='text-align:center;'>  Medical Insuarance Cost Estimator </h1>",unsafe_allow_html=True)
with st.form("smoker"):
    ag=st.text_input("AGE")
    weight=st.text_input("BMI")
    gender=st.text_input("SEX(male or female)")
    smoke=st.text_input("SMOKER(yes or no)")
    child=st.text_input("No. of children")
    area=st.text_input("Enter region(northeast,northwest,southwest,southeast)")
    status=st.form_submit_button("Submit")
if status==True and ag!="" and gender!=""and weight!="" and smoke!="" and area!="" and child!="":
    import pandas as pd 
    healthData=pd.read_csv("insurance.csv")
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    numeric_columns=healthData.select_dtypes(include=np.number).columns.tolist()[:-1]
    categorical_columns=healthData.select_dtypes(include='object').columns.tolist()
    from sklearn.preprocessing import OneHotEncoder
    cat=healthData[categorical_columns].fillna("unkown")
    enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    enc.fit(cat)
    enccols=list(enc.get_feature_names_out(categorical_columns))
    healthData[enccols]=enc.transform(cat)
    inputs=healthData[(numeric_columns + enccols)]#inpputs must be in 2D
    targets=healthData["charges"]
    data={"age":int(ag),"bmi":int(weight),"sex":gender,"smoker":smoke,"children":int(child),"region":area}
    newInputs=pd.DataFrame([data])
    New_Category=newInputs.select_dtypes(include='object').columns.tolist()
    New_Numeric=newInputs.select_dtypes(include=np.number).columns.tolist()
    enccol=list(enc.get_feature_names_out(New_Category))
    newInputs[enccol]=enc.transform(newInputs[New_Category])
    Ninputs=newInputs[(New_Numeric + enccol)]
    from sklearn.tree import DecisionTreeRegressor
    model=DecisionTreeRegressor()
    #training our model
    model.fit(inputs,targets)
    if int(ag) > 18:
        predict=model.predict(Ninputs)
        result=("Your Annual Insuarance Cost Should be around:")
        #getting em' predictions
        st.markdown(result)
        st.markdown(predict[0])
        st.markdown("NOTE: These charges do not include any underlying medical conditions..If you happen to have any underlying medical conditions, the charges will definately go higher.")
    else:
        st.warning("Age MUST be greater than or equal to 18!")
else:
    st.warning("Fill all the above fields!")
