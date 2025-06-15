import streamlit as st
st.markdown("<h3 style='text-align:center;'>  Medical Insuarance Cost Estimator </h1>",unsafe_allow_html=True)
with st.form("smoker"):
    ag=st.text_input("AGE")
    weight=st.text_input("BMI")
    gender=st.selectbox("SEX",options=("male","female"))
    smoke=st.selectbox("SMOKER",options=("yes","no"))
    child=st.text_input("No. of children")
    area=st.selectbox("Region",options=("northeast","northwest","southwest","southeast"))
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
    from sklearn.ensemble import GradientBoostingRegressor
    model=GradientBoostingRegressor(n_estimators=500,subsample=0.5,random_state=42,learning_rate=0.1)
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
import pandas as pd 
healthData=pd.read_csv("insurance.csv")
import matplotlib.pyplot as plt
with st.sidebar:
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
    from sklearn.ensemble import GradientBoostingRegressor
    model=GradientBoostingRegressor(n_estimators=500,subsample=0.5,random_state=42,learning_rate=0.1)
    #training our model
    model.fit(inputs,targets)
    st.markdown("## Visualizations")
    st.write("Feature importances/ weights")
    fig1=plt.figure()
    plt.barh(inputs.columns,model.feature_importances_)
    plt.title("Feature Weights")
    plt.ylabel("Feature")
    plt.xlabel("weight")
    st.write(fig1)
    st.write("Graphs of input features against the target feature")
    fig=plt.figure()
    plt.scatter(healthData.age,healthData.charges)
    plt.title("Age vs Charges")
    plt.ylabel("Charges")
    plt.xlabel("Age")
    st.write(fig)
    fig2=plt.figure()
    plt.scatter(healthData.bmi,healthData.charges)
    plt.title("BMI vs Charges")
    plt.ylabel("Charges")
    plt.xlabel("BMI")
    st.write(fig2)
    fig3=plt.figure()
    plt.bar(healthData.sex,healthData.charges)
    plt.title("Sex vs Charges")
    plt.ylabel("Charges")
    plt.xlabel("Sex")
    st.write(fig3)
    fig4=plt.figure()
    plt.bar(healthData.smoker,healthData.charges)
    plt.title("Smoker vs Charges")
    plt.ylabel("Charges")
    plt.xlabel("Smoker")
    st.write(fig4)
    fig5=plt.figure()
    plt.scatter(healthData.children,healthData.charges)
    plt.title("Children vs Charges")
    plt.ylabel("Charges")
    plt.xlabel("Children")
    st.write(fig5)
    fig6=plt.figure()
    plt.bar(healthData.region,healthData.charges)
    plt.title("Region vs Charges")
    plt.ylabel("Charges")
    plt.xlabel("Region")
    st.write(fig6)
    