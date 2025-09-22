system_message = """You are an expert in image restoration. Given an image of low quality, your task is guiding the user to utilize various tools to enhance its quality. The input image may suffer from various kinds of degradations, including low resolution, noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. The available tools each specialize in addressing one of the above eight degradations, i.e., super-resolution, denoising, motion deblurring, defocus deblurring, dehazing, deraining, brightening, and jpeg compression artifact removal. The following will be a continuation of an interaction between you and a user to restore an image. Note that if the user specifies the output format, you must strictly follow it without any other words."""

gpt_evaluate_degradation_prompt = """Here's an image to restore. Please assess it with respect to the following seven degradations: noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. For each degradation, please explicitly give your thought and the severity. Be as precise and concise as possible. Your output must be in the format of a list of JSON objects, each having three fields: "degradation", "thought", and "severity". "degradation" must be one of ["noise", "motion blur", "defocus blur", "haze", "rain", "dark", "jpeg compression artifact"]; "thought" is your thought on this degradation of the image; "severity" must be one of "very low", "low", "medium", "high", "very high". Here's a simple example of the format:
[
    {
        "degradation": "noise",
        "thought": "The image does not appear to be noisy.",
        "severity": "low"
    },
    {
        "degradation": "motion blur",
        "thought": "The image is blurry in the vertical direction, which is likely caused by motion of the camera.",
        "severity": "high"
    },
    {
        "degradation": "defocus blur",
        "thought": "The image does not seem to be out of focus.",
        "severity": "low"
    },
    {
        "degradation": "haze",
        "thought": "There is somewhat haze in the image.",
        "severity": "medium"
    },
    {
        "degradation": "rain",
        "thought": "There is no visible rain in the image.",
        "severity": "very low"
    },
    {
        "degradation": "dark",
        "thought": "The lighting in the image is bright.",
        "severity": "very low"
    },
    {
        "degradation": "jpeg compression artifact",
        "thought": "Blocking artifacts, ringing artifacts, and color bleeding are visible in the image, indicating jpeg compression artifact.",
        "severity": "very high"
    },
]"""

depictqa_evaluate_degradation_prompt = """What's the severity of {degradation} in this image? Answer the question using a single word or phrase in the followings: very low, low, medium, high, very high."""

distill_knowledge_prompt = """We are studying image restoration with multiple degradations. The degradation types we are focusing on are: low-resolution, noise, motion blur, defocus blur, rain, haze, dark, and jpeg compression artifact. We have tools to address these degradations, that is, we can conduct these tasks: super-resolution, denoising, motion deblurring, defocus deblurring, deraining, dehazing, brightening, and jpeg compression artifact removal. The problem is, given the tasks to conduct, we need to determine the order of them. This is very complicated because different tasks may have special requirements and side-effects, and the correct order of tasks can significantly affect the final result. We have conducted some trials and collected the following experience:
{experience}
Please distill knowledge from this experience that will be valuable for determining the order of tasks. Note that the degradations can be more complex than what we have encountered above."""

schedule_w_retrieval_prompt = """There's an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. Now we need to determine the order of these unordered tasks. For your information, based on past trials, we have the following experience:
{experience}
Based on this experience, please give the correct order of the tasks. Your output must be a JSON object with two fields: "thought" and "order", where "order" must be a permutation of {agenda} in the order you determine."""

reason_to_schedule_prompt = """There's an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. Please provide some insights into the correct order of these unordered tasks. You should pay special attention to the essence and side-effects of these tasks."""

schedule_wo_retrieval_prompt = """There's an image suffering from {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. To determine the order of them, we should consider that: 
{insights} 
Based on these insights, please give the correct order of the tasks. Your output must be a list of the tasks in the order you determine, which is a permutation of {agenda}."""

reschedule_ps_prompt = "\nBesides, in attempts just now, we found the result is unsatisfactory if {failed_tries} is conducted first. Remember not to arrange {failed_tries} in the first place."

gpt_evaluate_tool_result_prompt = """What's the severity of {degradation} in this image? Please provide your reasoning. Your output must be a JSON object with two fields: "thought" and "severity", where "severity" must be one of "very low", "low", "medium", "high", "very high"."""

gpt_compare_prompt = """Which of the two images do you consider to be of better quality? Please provide your reasoning. Your output must be a JSON object with two fields: "thought" and "choice", where "choice" is either "former" or "latter", indicating which image you think is of higher quality. An exception is that if you think the difference is negligible, you can choose "neither" as "choice"."""

depictqa_compare_prompt = "Which of the two images, Image A or Image B, do you consider to be of better quality? Answer the question using a single word or phrase."
