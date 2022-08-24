# Neural Network Regression with TensorFlow - Medical Insurance Premium Prediction
Medical Insurance Predictions

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Medical_Insurance_Premium_Prediction.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "##Importing all necessary libraries"
      ],
      "metadata": {
        "id": "iiQBWDH3B8Hd"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "JpBid9DSKzmq"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import zipfile as zp\n",
        "import os\n",
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Mounting Google Drive to Import Dataset from Google Drive"
      ],
      "metadata": {
        "id": "F0KxBWDAClnr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "import io\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "grNV3YqOLauz",
        "outputId": "f404f444-9cfc-4892-ea1e-6b5bfc4cfe57"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Importing DataSet"
      ],
      "metadata": {
        "id": "tpQFCXwmCv_Q"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Reading and saving the data \n",
        "insurance = pd.read_csv(\"Your File Path\")\n",
        "insurance"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 487
        },
        "id": "JsO13veOZzap",
        "outputId": "e3a93ee4-b1ba-417c-c6ec-dd93dd5fb44f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     Age  Diabetes  BloodPressureProblems  AnyTransplants  AnyChronicDiseases  \\\n",
              "0     45         0                      0               0                   0   \n",
              "1     60         1                      0               0                   0   \n",
              "2     36         1                      1               0                   0   \n",
              "3     52         1                      1               0                   1   \n",
              "4     38         0                      0               0                   1   \n",
              "..   ...       ...                    ...             ...                 ...   \n",
              "981   18         0                      0               0                   0   \n",
              "982   64         1                      1               0                   0   \n",
              "983   56         0                      1               0                   0   \n",
              "984   47         1                      1               0                   0   \n",
              "985   21         0                      0               0                   0   \n",
              "\n",
              "     Height  Weight  KnownAllergies  HistoryOfCancerInFamily  \\\n",
              "0       155      57               0                        0   \n",
              "1       180      73               0                        0   \n",
              "2       158      59               0                        0   \n",
              "3       183      93               0                        0   \n",
              "4       166      88               0                        0   \n",
              "..      ...     ...             ...                      ...   \n",
              "981     169      67               0                        0   \n",
              "982     153      70               0                        0   \n",
              "983     155      71               0                        0   \n",
              "984     158      73               1                        0   \n",
              "985     158      75               1                        0   \n",
              "\n",
              "     NumberOfMajorSurgeries  PremiumPrice  \n",
              "0                         0         25000  \n",
              "1                         0         29000  \n",
              "2                         1         23000  \n",
              "3                         2         28000  \n",
              "4                         1         23000  \n",
              "..                      ...           ...  \n",
              "981                       0         15000  \n",
              "982                       3         28000  \n",
              "983                       1         29000  \n",
              "984                       1         39000  \n",
              "985                       1         15000  \n",
              "\n",
              "[986 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-dfd4f1e2-86e6-4f03-8a87-18c9dc4daa82\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Diabetes</th>\n",
              "      <th>BloodPressureProblems</th>\n",
              "      <th>AnyTransplants</th>\n",
              "      <th>AnyChronicDiseases</th>\n",
              "      <th>Height</th>\n",
              "      <th>Weight</th>\n",
              "      <th>KnownAllergies</th>\n",
              "      <th>HistoryOfCancerInFamily</th>\n",
              "      <th>NumberOfMajorSurgeries</th>\n",
              "      <th>PremiumPrice</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>45</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>155</td>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>25000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>60</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>180</td>\n",
              "      <td>73</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>29000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>36</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>158</td>\n",
              "      <td>59</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>23000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>52</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>183</td>\n",
              "      <td>93</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>28000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>38</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>166</td>\n",
              "      <td>88</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>23000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>981</th>\n",
              "      <td>18</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>169</td>\n",
              "      <td>67</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>15000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>982</th>\n",
              "      <td>64</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>153</td>\n",
              "      <td>70</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>28000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>983</th>\n",
              "      <td>56</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>155</td>\n",
              "      <td>71</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>29000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>984</th>\n",
              "      <td>47</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>158</td>\n",
              "      <td>73</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>39000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>985</th>\n",
              "      <td>21</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>158</td>\n",
              "      <td>75</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>15000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>986 rows × 11 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-dfd4f1e2-86e6-4f03-8a87-18c9dc4daa82')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-dfd4f1e2-86e6-4f03-8a87-18c9dc4daa82 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-dfd4f1e2-86e6-4f03-8a87-18c9dc4daa82');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Visualizing Data"
      ],
      "metadata": {
        "id": "WF3lkDDQDDsA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Data values \n",
        "insurance.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 270
        },
        "id": "SsxEQJU8L_pv",
        "outputId": "5fb49c20-ab4d-4a9b-f00d-9548380360b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Age  Diabetes  BloodPressureProblems  AnyTransplants  AnyChronicDiseases  \\\n",
              "0   45         0                      0               0                   0   \n",
              "1   60         1                      0               0                   0   \n",
              "2   36         1                      1               0                   0   \n",
              "3   52         1                      1               0                   1   \n",
              "4   38         0                      0               0                   1   \n",
              "\n",
              "   Height  Weight  KnownAllergies  HistoryOfCancerInFamily  \\\n",
              "0     155      57               0                        0   \n",
              "1     180      73               0                        0   \n",
              "2     158      59               0                        0   \n",
              "3     183      93               0                        0   \n",
              "4     166      88               0                        0   \n",
              "\n",
              "   NumberOfMajorSurgeries  PremiumPrice  \n",
              "0                       0         25000  \n",
              "1                       0         29000  \n",
              "2                       1         23000  \n",
              "3                       2         28000  \n",
              "4                       1         23000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-4c4e6a8a-0175-4eed-88c5-289a4cac3dfa\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Diabetes</th>\n",
              "      <th>BloodPressureProblems</th>\n",
              "      <th>AnyTransplants</th>\n",
              "      <th>AnyChronicDiseases</th>\n",
              "      <th>Height</th>\n",
              "      <th>Weight</th>\n",
              "      <th>KnownAllergies</th>\n",
              "      <th>HistoryOfCancerInFamily</th>\n",
              "      <th>NumberOfMajorSurgeries</th>\n",
              "      <th>PremiumPrice</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>45</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>155</td>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>25000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>60</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>180</td>\n",
              "      <td>73</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>29000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>36</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>158</td>\n",
              "      <td>59</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>23000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>52</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>183</td>\n",
              "      <td>93</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>28000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>38</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>166</td>\n",
              "      <td>88</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>23000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-4c4e6a8a-0175-4eed-88c5-289a4cac3dfa')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-4c4e6a8a-0175-4eed-88c5-289a4cac3dfa button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-4c4e6a8a-0175-4eed-88c5-289a4cac3dfa');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Normalization and Standadization of Data"
      ],
      "metadata": {
        "id": "Oh1aGzHHDHl4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.compose import make_column_transformer\n",
        "from sklearn.preprocessing import MinMaxScaler,OneHotEncoder\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "#Create a Column Transformer\n",
        "\n",
        "ct = make_column_transformer(\n",
        "    (MinMaxScaler(),[\"Age\",\"Height\",\"Weight\",\"NumberOfMajorSurgeries\"]),\n",
        "    (OneHotEncoder(handle_unknown = \"ignore\"),[\"Diabetes\",\"BloodPressureProblems\",\"AnyTransplants\",\"AnyChronicDiseases\",\"KnownAllergies\",\"HistoryOfCancerInFamily\"])\n",
        ")\n",
        "#Creating X and Y values\n",
        "x = insurance.drop(\"PremiumPrice\",axis = 1)\n",
        "y = insurance[\"PremiumPrice\"]\n",
        "\n",
        "#Create trainning and test set(use scikit)\n",
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)#20% test size\n",
        "\n",
        "# Fit the column transformer to our training data\n",
        "ct.fit(x_train)\n",
        "\n",
        "#Transform trainning and test data normalization (MinMaxScaler) and one Hot\n",
        "x_train_normal = ct.transform(x_train) \n",
        "x_test_normal = ct.transform(x_test)"
      ],
      "metadata": {
        "id": "zRlIrKZTME09"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train_normal.shape,x_train.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vLrzUS6WRi4D",
        "outputId": "566d967c-bb76-4cdb-a1f9-fababd117bf2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((788, 16), (788, 10))"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Steps in modelling with TensorFlow\n",
        "\n",
        "`1.) Creating a model` - define the input and output layers, as well as the hidden layers of a deep learning model.\n",
        "\n",
        "`2.) Compiling a model` - define the loss funtion (in other words, the function which tells our model how wrong it is) and the optimizer (tells our model how to improve the patterns its learning) and evaluation metrics (what we can use to interpret the performance of our model).\n",
        "\n",
        "`3.) Fitting a model` - letting the model try to find patterns between X & y (features and labels)."
      ],
      "metadata": {
        "id": "JXuhRBrgDac2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Building a Neural Network\n",
        "tf.random.set_seed(42)\n",
        "\n",
        "#1. Create the model\n",
        "model_insurance = tf.keras.Sequential([\n",
        "    tf.keras.layers.Dense(100,activation=\"relu\"),\n",
        "    tf.keras.layers.Dense(10,activation=\"relu\"),\n",
        "    tf.keras.layers.Dense(1)\n",
        "])\n",
        "\n",
        "#2. Compile Model\n",
        "model_insurance.compile(\n",
        "    loss = tf.keras.losses.mae,\n",
        "    optimizer = tf.keras.optimizers.Adam(),\n",
        "    metrics = [\"mae\"]\n",
        ")\n",
        "\n",
        "#3. Fitting the model\n",
        "model_history = model_insurance.fit(\n",
        "    x_train_normal,\n",
        "    y_train,\n",
        "    epochs = 450,\n",
        "    verbose=0\n",
        ")"
      ],
      "metadata": {
        "id": "GB5qgnuJOD4w"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Plotting History Graph"
      ],
      "metadata": {
        "id": "R8PgTprGEMt_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#plot history (also known as the loss curve)\n",
        "import matplotlib.pyplot as plt\n",
        "pd.DataFrame(model_history.history).plot()\n",
        "plt.ylabel(\"Loss\")\n",
        "plt.xlabel(\"Epochs\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "id": "_ouOpRvdTLJS",
        "outputId": "cb6170ec-9eb4-44ea-fe7f-261a27960e3c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 0, 'Epochs')"
            ]
          },
          "metadata": {},
          "execution_count": 8
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "evaluated_value = model_insurance.evaluate(x_test_normal,y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z7KOIPGQTSZv",
        "outputId": "c958e1e3-f853-412f-dd0c-28a32ec51a81"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "7/7 [==============================] - 0s 3ms/step - loss: 2721.8801 - mae: 2721.8801\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model_insurance.predict(x_test_normal)"
      ],
      "metadata": {
        "id": "vVuIAlbVUGEf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pd.DataFrame(model_history.history).plot()"
      ],
      "metadata": {
        "id": "pAk0Ny9LT22b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 284
        },
        "outputId": "379c3a09-e4fa-4196-e292-f25715f03418"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f9df021e250>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model_insurance.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SWrNH-GwkfYW",
        "outputId": "f979786f-f06d-4c49-fb45-1d82623cc646"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " dense (Dense)               (None, 100)               1700      \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 10)                1010      \n",
            "                                                                 \n",
            " dense_2 (Dense)             (None, 1)                 11        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 2,721\n",
            "Trainable params: 2,721\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "gH9Lo6n6NLTF"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
