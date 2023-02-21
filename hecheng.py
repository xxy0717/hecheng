import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# 设置页面标题和头部
st.set_page_config(page_title='孩子头像生成器', page_icon=':baby:', layout='wide')

st.title("孩子头像生成器")

# 定义上传照片函数
def upload_image(name):
    uploaded_image = st.file_uploader(f"上传{name}照片", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption=f"{name}照片", use_column_width=True)
        return image

# 上传父亲和母亲的照片
father_image = upload_image("父亲")
mother_image = upload_image("母亲")

# 如果上传了两张照片
if father_image is not None and mother_image is not None:

    # 调整父亲和母亲的图像大小和模式
    father_image = father_image.convert('RGB').resize((256, 256))
    mother_image = mother_image.convert('RGB').resize((256, 256))

    # 按比例合成孩子头像
    alpha = np.random.uniform(0.2, 0.8)  # 随机选择 alpha 值
    child_array = np.array(alpha * np.array(father_image) + (1 - alpha) * np.array(mother_image), dtype=np.uint8)
    child_image = Image.fromarray(child_array)

    # 显示合成的孩子头像
    st.header("合成的孩子头像：")
    st.image(child_image, caption="孩子照片", use_column_width=True)
