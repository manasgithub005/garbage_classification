import tensorflow as tf
import numpy as np

model = None
output_class = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
data = {
"cardboard":
	["Cardboard, also referred to as corrugated cardboard, is a recyclable material that is recycled by small and large scale businesses to save money on waste disposal costs. Cardboard recycling is the reprocessing and reuse of thick sheets or stiff multilayered papers that have been used, discarded or regarded as waste.<br><br>Cardboard boxes are usually heavy-duty or thick-sheets of paper known for their durability and hardness. Examples of cardboard include packaging boxes, egg cartons, shoe boxes, and cereal boxes.Recycling is good for us as it not only saves our environment from deterioration by reducing pollution but also conserves valuable resources and creates jobs. Cardboard recycling is done as a way of keeping the environment clean and green. The steps below provide an explanation of the cardboard recycling system.",
	"4XOAGNzWvqY", "oKFOqMZmuA8"],	
"glass":
	["Glass recycling is the processing of waste glass into usable products. Glass that is crushed and ready to be remelted is called cullet. There are two types of cullet: internal and external. Internal cullet is composed of defective products detected and rejected by a quality control process during the industrial process of glass manufacturing, transition phases of product changes (such as thickness and colour changes) and production offcuts. External cullet is waste glass that has been collected or reprocessed with the purpose of recycling. External cullet (which can be pre- or post-consumer) is classified as waste. The word \"cullet\", when used in the context of end-of-waste, will always refer to external cullet.<br><br>To be recycled, glass waste needs to be purified and cleaned of contamination. Then, depending on the end use and local processing capabilities, it might also have to be separated into different colors. Many recyclers collect different colors of glass separately since glass retains its color after recycling.",
	"bYVih298o1Y", "6R8YObQbE88"],
"metal":
	["Several kinds and also large amounts of metals are used in industrial processes every day. Since the industrial revolution period has taken place, our consumption levels skyrocketed due to the mass production of goods and the resulting low unit price.<br><br>The most consumed metal worldwide is aluminum, followed by copper, zinc, lead and nickel. Moreover, some precious materials like gold are used for our computers and other electronic devices.<br><br>Metal is therefore crucial to sustaining our living standard. However, metals are resources that are limited. The depletion of metals can be a big issue in the future since the world population grows rapidly and thus also the demand for goods made out of metal will increase.<br><br>To mitigate the problem of metal depletion, we have to look out for effective measures. One of those measures could be metal recycling.",
	"qAGCI0-pQ3E", "rgEEXhbar3A"],
"paper":
	["The recycling of paper is the process by which waste paper is turned into new paper products. It has a number of important benefits: It saves waste paper from occupying homes of people and producing methane as it breaks down. Because paper fibre contains carbon (originally absorbed by the tree from which it was produced), recycling keeps the carbon locked up for longer and out of the atmosphere. Around two-thirds of all paper products in the US are now recovered and recycled, although it does not all become new paper. After repeated processing the fibres become too short for the production of new paper - this is why virgin fibre (from sustainably farmed trees) is frequently added to the pulp recipe.<br><br>Paper recycling pertains to the processes of reprocessing waste paper for reuse. Waste papers are either obtained from paper mill paper scraps, discarded paper materials, and waste paper material discarded after consumer use. Examples of the commonly known papers recycled are old newspapers and magazines.",
	"jAqVxsEgWIM", "xhW0RTg8kRI"],
"plastic":
	["Plastic recycling is the process of recovering scrap or waste plastic and reprocessing the material into useful products. Due to purposefully misleading symbols on plastic packaging and numerous technical hurdles, less than 10% of plastic has ever been recycled. Compared with the lucrative recycling of metal, and similar to the low value of glass recycling, plastic polymers recycling is often more challenging because of low density and low value.<br><br>Materials recovery facilities are responsible for sorting and processing plastics. As of 2019, due to limitations in their economic viability, these facilities have struggled to make a meaningful contribution to the plastic supply chain. The plastics industry has known since at least the 1970s that recycling of most plastics is unlikely because of these limitations. However, the industry has lobbied for the expansion of recycling while these companies have continued to increase the amount of virgin plastic being produced.",
	"rYwBL_6hB2I", "I_fUpP-hq3A"],
"trash":
    ["Garbage, trash, rubbish, or refuse is waste material that is discarded by humans, usually due to a perceived lack of utility. The term generally does not encompass bodily waste products, purely liquid or gaseous wastes, or toxic waste products. Garbage is commonly sorted and classified into kinds of material suitable for specific kinds of disposal.<br><br>In urban areas, garbage of all kinds is collected and treated as municipal solid waste; garbage that is discarded in ways that cause it to end up in the environment, rather than in containers or facilities designed to receive garbage, is considered litter. Litter is a form of garbage that has been improperly disposed of, and which therefore enters the environment.[8] Notably, however, only a small fraction of garbage that is generated becomes litter, with the vast majority being disposed of in ways intended to secure it from entering the environment.",
	"Bhi7S06pwv4", "IHPBJySIXZw"],
	
}


def load_artifacts():
    global model
    model = tf.keras.models.load_model("classifyWaste.h5")

def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value, data[predicted_value][0], data[predicted_value][1], data[predicted_value][2]